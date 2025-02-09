import streamlit as st
import tempfile
import os
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import *


# API GROC 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

TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'PDF', 'CSV', 'TXT'
]

CONFIG_MODELS = {
    'Groc':
    {'modelos': ['gemma2-9b-it', 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768'],
     'chat': ChatGroq},
    'OpenAI':
    {'modelos': ['o1-mini', 'gpt-4o'],
     'chat': ChatOpenAI}
}

MEMORIA = ConversationBufferMemory()


def carrega_arquivo(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        documento = carrega_site(arquivo)
    if tipo_arquivo == 'Youtube':
        documento = carrega_youtube(arquivo)
    if tipo_arquivo == 'PDF':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)
        print(documento)
    if tipo_arquivo == 'CSV':
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)
    if tipo_arquivo == 'TXT':
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)
    return documento


def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):

    documento = carrega_arquivo(tipo_arquivo, arquivo)

    system_message = '''Você é um representante de uma empresa farmacêutica e tem a função de explicar para o médico as vantagens de prescrever os medicamentos que voce promove, 
    seu linguajar é técnico, respeitoso mas no fundo tem sempre uma saida amigável fazendo uma conexão com o médico para que ele te veja como um amigo.
    Você possui acesso às seguintes informações vindas de um documento {}:
    ####
    {}
    ####
    Utilize as informações fornecidas para basear as suas respostas.
    Sempre que houver $ na sua saída, substita por S.
    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue"
    sugira ao usuário carregar novamente o Oráculo!'''.format(tipo_arquivo, documento)

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    chat = CONFIG_MODELS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat
    st.session_state['chain'] = chain


def pagina_chat():
    st.header('🤖 Bem-vindo ao ReppChat 🤖', divider=True)
    chain = st.session_state.get('chain')

    if chain is None:
        st.error('Carregue o Oráculo')
        st.stop

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Digite aqui sua pergunta')
    if input_usuario:
        # input do usuario humano
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        # resposta da IA com texto digitado
        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input': input_usuario,
            'chat_history': memoria.buffer_as_messages}))

        # coloca as mensagens na memoria
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria


def sidebar():
    tabs = st.tabs(['Upload de arquivos', 'Seleção de modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox(
            'Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a URL do site')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a URL do video')
        if tipo_arquivo == 'PDF':
            arquivo = st.file_uploader(
                'Faça aqui o upload do arquivo', type=['.pdf'])
        if tipo_arquivo == 'CSV':
            arquivo = st.file_uploader(
                'Faça aqui o upload do arquivo', type=['.csv'])
        if tipo_arquivo == 'TXT':
            arquivo = st.file_uploader(
                'Faça aqui o upload do arquivo', type=['.txt'])
    with tabs[1]:
        provedor = st.selectbox(
            'Selecione o provedor de IA', CONFIG_MODELS.keys())
        modelo = st.selectbox('Selecione o modelo de IA',
                              CONFIG_MODELS[provedor]['modelos'])

     # criado uma session state para que o valor default seja sempre o ultimo digitado
        if provedor == 'OpenAI':
            api_key = st.text_input(
                f'Adicione a chave API para o provedor {provedor}',
                value=st.session_state.get(f'api_key_{provedor}'))
            st.session_state[f'api_key_{provedor}'] = api_key

        if provedor == 'Groc':
            api_key = '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'
            st.session_state[f'api_key_{api_key}'] = api_key

    if st.button('Inicializar ReppChat', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA


def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()
