# IMPORTAÇÕES #
from diagno_train import (
    carregar_doencas,
    preparar_base,
    diagnosticar,
    normalizar
)

# Biblioteca para o modelo
from gensim.models import FastText

caminho_json = "data/medical_data.json"
# FUNÇÃO PRINCIPAL #

"""
    Função Responsável por:
        - carregar dados
        - treinar modelo
        - preparar base
        - interagir com o usuário
"""
def main():

    print("=== Sistema de Diagnóstico Inteligente ===\n")

    # 1. EXCEPTION RESPONSAVEL POR CARREGAR A BASE DE DOENÇAS #

    try:
        doencas = carregar_doencas(caminho_json)
    except Exception as e:
        print("Erro ao carregar base de dados:", e)
        return

    # 2. PREPARAR CORPUS PARA TREINO #

    corpus = []

    """
        Loop para preparar o corpus de treino.
        - Para cada doença na base de dados:
            - Junta os sintomas em uma única string
            - Normaliza a string usando a função normalizar, que remove acentos, pontuação e converte para minúsculas
            - Adiciona a lista de tokens normalizados ao corpus, que será usado para treinar o modelo FastText
    """
    for d in doencas:
        texto = " ".join(d["sintomas"])
        tokens = normalizar(texto)
        corpus.append(tokens)

    # 3. TREINAR MODELO FASTTEXT #

    print("Treinando modelo NLP...")

    model = FastText(
        sentences=corpus,
        vector_size=100,   # dimensão dos vetores
        window=3,          # contexto
        min_count=1        # incluir todas palavras (dataset pequeno)
    )

    print("Modelo treinado com sucesso!\n")

    # 4. PREPARAR BASE VETORIAL #

    """
        Converte todas doenças em vetores usando o modelo treinado
        - A função preparar_base cria uma lista de tuplas (doença, vetor) para facilitar o diagnóstico posterior

    """
    base_vetores = preparar_base(model, doencas)

    # 5. LOOP DE INTERAÇÃO #

    while True:
        print("\nDigite seus sintomas (ou 'sair' para encerrar):")
        entrada = input(">> ")

        # 5.1 SAÍDA DO SISTEMA #

        if entrada.lower() in ["sair", "exit", "quit"]:
            print("Encerrando sistema...")
            break

        # 5.2 VALIDAR INPUT #

        if not entrada.strip():
            print("Por favor, descreva seus sintomas.")
            continue

        # 5.3 EXECUTAR DIAGNÓSTICO #

        resultados = diagnosticar(
            model,
            base_vetores,
            entrada,
            top_n=3  # top 3 diagnósticos
        )

        # 5.4 MOSTRAR RESULTADOS #

        print("\nPossíveis diagnósticos:\n")

        for i, (doenca, score) in enumerate(resultados, start=1):
            print(f"{i}. Doença: {doenca['doenca']}")
            print(f"   Score de similaridade: {score:.4f}")
            print(f"   Sintomas: {', '.join(doenca['sintomas'])}")
            print(f"   Prescrição: {doenca.get('tratamento', 'N/A')}")
            print("-" * 40)

        print(f"   \nLembrar que este é um diagnóstico preliminar e não substitui uma consulta médica. \nProcure um profissional de saúde para avaliação completa.\n")
        for i in range(1):
            print("-" * 40)


# EXECUÇÃO #

# Garante que o código só roda diretamente (não quando importado)
if __name__ == "__main__":
    main()