import pandas as pd


#######################################
#       CRIAR FLAGS BINÁRIAS         #
#######################################
import pandas as pd

def criar_flags(df: pd.DataFrame
) -> pd.DataFrame:
    """
    Cria Flags para variáveis relacionadas a riscos financeiros e retorna um DataFrame com as novas colunas.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados originais.

    Retorno:
    --------
    pd.DataFrame
        O DataFrame original com novas colunas binárias adicionadas.

    Indicadores Criados:
    --------------------
    - "flag_acao_judicial" → 1 se há ações judiciais, 0 caso contrário.
    - "flag_protesto" → 1 se há protestos, 0 caso contrário.
    - "flag_dividas_vencidas" → 1 se há dívidas vencidas, 0 caso contrário.
    - "flag_valor_vencido" → 1 se há valores vencidos, 0 caso contrário.
    - "flag_valor_por_vencer" → 1 se há valores a vencer, 0 caso contrário.
    - "flag_valor_quitado" → 1 se já houve pagamentos, 0 caso contrário.
    """

    df = df.copy()  # Para evitar modificações no DataFrame original

    # Criar indicadores binários
    df["flag_acao_judicial"] = (df["quant_acao_judicial"] > 0).astype(int)
    df["flag_protesto"] = (df["quant_protestos"] > 0).astype(int)
    df["flag_dividas_vencidas"] = (df["dividas_vencidas_qtd"] > 0).astype(int)
    df["flag_valor_vencido"] = (df["valor_vencido"] > 0).astype(int)
    df["flag_valor_por_vencer"] = (df["valor_por_vencer"] > 0).astype(int)
    df["flag_valor_quitado"] = (df["valor_quitado"] > 0).astype(int)

    return df



#######################################
#       AGRUPAR PRAZO DE PAGAMENTO    #
#######################################
import pandas as pd

def agrupar_prazo(df: pd.DataFrame, forma_pgto: str) -> pd.DataFrame:
    """
    Agrupa as formas de pagamento por prazo médio e retorna um DataFrame com os valores agrupados.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados a serem processados.
    forma_pgto : str
        Nome da coluna que contém as formas de pagamento.

    Retorno:
    --------
    pd.DataFrame
        O DataFrame original com uma nova coluna "prazo_pagamento" contendo os valores agrupados.
    
    Agrupamentos:
    -------------
    - "Sem pagamento" → Quando a informação indica explicitamente a ausência de pagamento.
    - "Desconhecido" → Quando o valor está ausente (nan, vazio, "none").
    - "À vista (até 15 dias)" → Pagamentos em uma única parcela em até 15 dias.
    - "Curto prazo (16-30 dias)" → Pagamentos com prazo médio entre 16 e 30 dias.
    - "Médio prazo (31-90 dias)" → Pagamentos com prazo médio entre 31 e 90 dias.
    - "Longo prazo (+90 dias)" → Pagamentos com prazo médio superior a 90 dias.
    - "Outros" → Casos em que não foi possível identificar o prazo.
    """

    def classificar_prazo(fp):
        # Verificar se o valor está ausente (nan, "", "none")
        if pd.isna(fp) or str(fp).strip().lower() in ["nan", "", "none"]:
            return "Desconhecido"
        
        # Verificar se o pagamento é explicitamente ausente
        if str(fp).strip().lower() in ["nenhum", "sem pagamento"]:
            return "Sem pagamento"

        # Converter para string e substituir "x" por "/" para padronizar
        prazos = [int(x) for x in str(fp).replace("x", "/").split("/") if x.isdigit()]

        # Se não houver prazos identificáveis, classificar como "Outros"
        if not prazos:
            return "Outros"

        # Calcular o prazo médio
        prazo_medio = sum(prazos) / len(prazos)

        # Definir categorias de prazo
        if len(prazos) == 1 and prazo_medio <= 15:
            return "À vista (até 15 dias)"
        elif prazo_medio <= 30:
            return "Curto prazo (16-30 dias)"
        elif prazo_medio <= 90:
            return "Médio prazo (31-90 dias)"
        else:
            return "Longo prazo (+90 dias)"
    
    # Criar e retornar nova coluna no DataFrame com os valores agrupados
    
    return df[forma_pgto].apply(classificar_prazo)




#######################################
#       AGRUPAR TIPO DE SOCIEDADE     #
#######################################
import pandas as pd

def agrupar_tipo_sociedade(df: pd.DataFrame, 
                           tipo_sociedade: str
) -> pd.DataFrame:
    """
    Agrupa os tipos de sociedade em categorias mais amplas e retorna um DataFrame com os valores agrupados.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados a serem processados.
    tipo_sociedade : str
        Nome da coluna que contém os tipos de sociedade.

    Retorno:
    --------
    pd.DataFrame
        O DataFrame original com uma nova coluna "{tipo_sociedade}_agrup" contendo os valores agrupados.

    Agrupamentos:
    -------------
    - "Sociedade Limitada" → Empresas LTDA.
    - "Empresa Individual" → Empresários individuais.
    - "Microempreendedor (MEI)" → Pequenos empresários MEI.
    - "Sociedade Anônima" → Empresas S.A.
    - "Cooperativas & Associações" → Cooperativas, associações e sindicatos.
    - "Organizações Públicas & Fundações" → Municípios, fundações, sociedades mistas, religiosas.
    - "Outros" → Tipos não categorizados.
    - "Desconhecido" → Valores ausentes (nan, "", "none").
    """

    def classificar_sociedade(tipo):
        # Verificar se o valor está ausente (nan, "", "none")
        if pd.isna(tipo) or str(tipo).strip().lower() in ["nan", "", "none"]:
            return "Desconhecido"

        # Normalizar para minúsculo
        tipo = str(tipo).strip().lower()

        # Classificar os tipos de sociedade
        if "limitada" in tipo:
            return "Sociedade Limitada"
        elif "individual" in tipo:
            return "Empresa Individual"
        elif "mei" in tipo:
            return "Microempreendedor (MEI)"
        elif "anonima" in tipo:
            return "Sociedade Anônima"
        elif "cooperativa" in tipo or "associacao" in tipo or "sindical" in tipo:
            return "Cooperativas & Associações"
        elif "municipio" in tipo or "fundacao" in tipo or "mista" in tipo or "religiosa" in tipo:
            return "Organizações Públicas & Fundações"
        else:
            return "Outros"

      # Criar nova coluna no DataFrame com os valores agrupados
    return df[tipo_sociedade].apply(classificar_sociedade)





#######################################
#       AGRUPAR ATIVIDADE PRINCIPAL   #
#######################################
import pandas as pd

def agrupar_atividade_principal(df: pd.DataFrame, 
                                atividade_col: str
) -> pd.DataFrame:
    """
    Agrupa as atividades principais em categorias mais amplas e retorna um DataFrame com os valores agrupados.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados a serem processados.
    atividade_col : str
        Nome da coluna que contém as atividades principais.

    Retorno:
    --------
    pd.DataFrame
        O DataFrame original com uma nova coluna "{atividade_col}_agrup" contendo os valores agrupados.

    Agrupamentos:
    -------------
    - "Comércio" → Comércio varejista, atacadista e distribuição.
    - "Indústria" → Fábricas e produção de bens.
    - "Serviços" → Consultoria, tecnologia, advocacia, saúde, transporte.
    - "Construção Civil" → Obras, engenharia e arquitetura.
    - "Agronegócio" → Agricultura, pecuária, produção rural.
    - "Educação" → Escolas, cursos, treinamentos.
    - "Saúde" → Clínicas médicas, hospitais, farmácias.
    - "Outros" → Tipos não categorizados.
    - "Desconhecido" → Valores ausentes (nan, "", "none").
    """

    def classificar_atividade(atividade):
        # Verificar se o valor está ausente (nan, "", "none")
        if pd.isna(atividade) or str(atividade).strip().lower() in ["nan", "", "none"]:
            return "Desconhecido"

        # Normalizar para minúsculo
        atividade = str(atividade).strip().lower()

        # Classificar as atividades em grandes grupos
        if "comércio" in atividade or "varejo" in atividade or "atacado" in atividade or "distribuição" in atividade or 'com de' in atividade:
            return "Comércio"
        elif "indústria" in atividade or "fábrica" in atividade or "produção" in atividade or 'ind de' in atividade:
            return "Indústria"
        elif "serviço" in atividade or "consultoria" in atividade or "tecnologia" in atividade or "transporte" in atividade or "advocacia" in atividade:
            return "Serviços"
        elif "construção" in atividade or "engenharia" in atividade or "arquitetura" in atividade:
            return "Construção Civil"
        elif "agricultura" in atividade or "pecuária" in atividade or "rural" in atividade:
            return "Agronegócio"
        elif "educação" in atividade or "escola" in atividade or "curso" in atividade or "treinamento" in atividade:
            return "Educação"
        elif "saúde" in atividade or "hospital" in atividade or "clínica" in atividade or "farmácia" in atividade:
            return "Saúde"
        elif 'entidades sem fins lucrativos' in atividade:
            return 'OSC'
        elif 'transporte' in atividade:
            return "Transporte"
        elif 'serv de selecao e administracao de pessoal' in atividade:
            return 'RH'
        else:
            return "Outros"

    # retornar nova coluna no DataFrame com os valores agrupados
    return df[atividade_col].apply(classificar_atividade)

#############################################
#     AGRUPAR TIPOS DE COMÉRCIO             #
#############################################

def agrupar_tipo_comercio(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Agrupa os tipos de comércio por categorias similares e retorna um DataFrame atualizado.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados a serem processados.
    coluna : str
        Nome da coluna que contém as descrições do comércio.

    Retorno:
    --------
    pd.DataFrame
        O DataFrame original com uma nova coluna "{coluna}_agrup" contendo os valores agrupados.

    Agrupamentos:
    -------------
    - "Tecnologia & Eletrônicos" → Equipamentos de informática, componentes eletrônicos, telefonia.
    - "Livros & Papelaria" → Livros, revistas, jornais.
    - "Móveis & Decoração" → Móveis, artigos de decoração, estofados.
    - "Moda & Vestuário" → Calçados, roupas, tecidos, acessórios.
    - "Automotivo" → Autopeças, pneus, motocicletas, tratores, veículos.
    - "Saúde & Ortopedia" → Produtos médicos, odontológicos, hospitalares.
    - "Construção & Ferragens" → Materiais de construção, ferramentas, ferragens.
    - "Alimentação & Bebidas" → Pães, doces, bebidas, produtos alimentícios.
    - "Esportes & Lazer" → Produtos esportivos, brinquedos, bicicletas.
    - "Outros" → Tipos não categorizados.
    """

    def classificar_comercio(comercio):
        # Verificar se o valor está ausente
        if pd.isna(comercio) or str(comercio).strip().lower() in ["nan", "", "none"]:
            return "Desconhecido"

        # Normalizar para minúsculo
        comercio = str(comercio).strip().lower()

        # Classificação por similaridade
        if "informatica" in comercio or "eletron" in comercio or "telefones" in comercio:
            return "Tecnologia & Eletrônicos"
        elif "livros" in comercio or "revistas" in comercio or "jornais" in comercio:
            return "Livros & Papelaria"
        elif "moveis" in comercio or "decoracao" in comercio or "estofados" in comercio:
            return "Móveis & Decoração"
        elif "confeccoes" in comercio or "calcados" in comercio or "tecidos" in comercio:
            return "Moda & Vestuário"
        elif "auto pecas" in comercio or "pneus" in comercio or "motocicletas" in comercio or "tratores" in comercio or "veiculos" in comercio:
            return "Automotivo"
        elif "medico" in comercio or "hospitalar" in comercio or "ortopedicos" in comercio or "odontologicos" in comercio:
            return "Saúde & Ortopedia"
        elif "construcao" in comercio or "ferragens" in comercio or "tintas" in comercio:
            return "Construção & Ferragens"
        elif "alimentos" in comercio or "bebidas" in comercio or "paes" in comercio or "doces" in comercio:
            return "Alimentação & Bebidas"
        elif "esportivos" in comercio or "brinquedos" in comercio or "bicicletas" in comercio:
            return "Esportes & Lazer"
        else:
            return "Outros"

    # Retornar df com os valores novos
    return df[coluna].apply(classificar_comercio)



#######################################
#       AGRUPAR OPÇÃO TRIBUTÁRIA      #
#######################################
def agrupar_opcao_tributaria(df: pd.DataFrame, 
                             opcao_tributaria: str
) -> pd.DataFrame:
    """
    Agrupa as opções tributárias em categorias mais amplas e retorna um DataFrame com os valores agrupados.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados a serem processados.
    opcao_tributaria : str
        Nome da coluna que contém as opções tributárias.

    Retorno:
    --------
    pd.DataFrame
        O DataFrame original com uma nova coluna "{opcao_tributaria}_agrup" contendo os valores agrupados.

    Agrupamentos:
    -------------
    - "Simples Nacional" → Empresas optantes pelo regime simplificado.
    - "Lucro Presumido" → Empresas tributadas com base na presunção do lucro.
    - "Lucro Real" → Empresas tributadas com base no lucro efetivo.
    - "MEI" → Microempreendedor Individual.
    - "Outros" → Tipos não categorizados.
    - "Desconhecido" → Valores ausentes (nan, "", "none").
    """

    def classificar_opcao_tributaria(opcao):
        # Verificar se o valor está ausente (nan, "", "none")
        if pd.isna(opcao) or str(opcao).strip().lower() in ["nan", "", "none"]:
            return "Desconhecido"

        # Normalizar para minúsculo
        opcao = str(opcao).strip().lower()

        # Classificar as opções tributárias
        if "simples" in opcao:
            return "Simples Nacional"
        elif "presumido" in opcao:
            return "Lucro Presumido"
        elif "real" in opcao:
            return "Lucro Real"
        elif "mei" in opcao:
            return "MEI"
        else:
            return "Outros"

    # Retorna o DataFrame com os valores agrupados
    return df[opcao_tributaria].apply(classificar_opcao_tributaria)
    





#######################################
#       Tratamento de Categóricas     #
#######################################

def tratar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata variáveis categóricas em um DataFrame, preenchendo valores nulos com "Desconhecido"
    e convertendo-as para valores numéricos com LabelEncoder.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo as variáveis categóricas.

    Retorno:
    --------
    pd.DataFrame
        DataFrame atualizado com os valores categóricos preenchidos e codificados.
    """
    df = df.copy()

    # Preenche valores nulos com "Desconhecido"
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Desconhecido")

    # Converte variáveis categóricas para numéricas
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Salva os encoders para referência futura, se necessário

    return df



#############################################
#      CRIAR VARIÁVEIS COMBINADAS          #
#############################################
import pandas as pd

def criar_variaveis_combinadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas variáveis combinadas para melhorar a modelagem preditiva.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados originais.

    Retorno:
    --------
    pd.DataFrame
        O DataFrame atualizado com novas variáveis criadas.

    Variáveis Criadas:
    --------------------
    - "razao_valor_vencido" → Razão entre valor vencido e valor total pago.
    - "razao_valor_quitado" → Razão entre valor quitado e valor total do pedido.
    - "media_ioi" → Média dos intervalos entre pedidos recentes.
    - "soma_riscos" → Soma total de ações judiciais, protestos e dívidas vencidas.
    - "historico_pagamento" → Proporção de valor quitado em relação ao total vencido e pago.
    """

    df = df.copy()  # Criar uma cópia para evitar modificar o original

    # Criar razões financeiras (evita divisão por zero adicionando +1)
    df["razao_valor_vencido"] = df["valor_vencido"] / (df["valor_quitado"] + 1)
    df["razao_valor_quitado"] = df["valor_quitado"] / (df["valor_total_pedido"] + 1)

    # Criar média dos intervalos entre pedidos
    df["media_ioi"] = (df["ioi_3months"] + df["ioi_36months"]) / 2

    # Criar soma total de riscos financeiros e jurídicos
    df["soma_riscos"] = df["quant_acao_judicial"] + df["quant_protestos"] + df["dividas_vencidas_qtd"]

    # Criar histórico de pagamento como proporção de valores pagos em relação ao vencido
    df["historico_pagamento"] = df["valor_quitado"] / (df["valor_quitado"] + df["valor_vencido"] + 1)

    return df



#############################################
#          ENCONTRAR CATEGÓRICAS            #
#############################################

def identificar_variaveis_categoricas(df: pd.DataFrame, threshold: int = 30) -> list:
    """
    Identifica automaticamente as variáveis categóricas no DataFrame.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    threshold : int, opcional (default=30)
        Número máximo de valores únicos para considerar uma variável numérica como categórica.

    Retorno:
    --------
    list
        Lista com os nomes das variáveis categóricas.
    """

    # Selecionar colunas do tipo 'object' ou 'category'
    categorias_explicit = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Identificar colunas numéricas com baixa cardinalidade (podem ser categóricas)
    categorias_numericas = [col for col in df.select_dtypes(include=['number']).columns
                            if df[col].nunique() <= threshold]

    # Unir ambas as listas
    variaveis_categoricas = list(set(categorias_explicit + categorias_numericas))

    return variaveis_categoricas



