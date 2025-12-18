"""
Painel de Controle Financeiro Pessoal
------------------------------------

Este script implementa um painel financeiro básico utilizando Streamlit. O objetivo é
replicar, de forma simplificada, as funcionalidades descritas na especificação do
sistema. Ele permite ao usuário importar extratos bancários, normalizar e
classificar transações por categoria, detectar transferências internas, e
visualizar resumos das receitas e despesas de cada ciclo financeiro.

Recursos principais:
* Upload de arquivo (CSV ou Excel) com transações.
* Normalização de colunas (data, descrição, valor).
* Remoção de duplicidades.
* Categorização automática baseada em palavras‑chave.
* Definição de ciclos mensais customizados (dia de início e fim).
* Resumo interativo das entradas e saídas, fixos vs variáveis, eventos
  extraordinários e investimentos.
* Detecção básica de transferências internas.
* Opção de exportar/atualizar dados em uma planilha Google Sheets (requer
  configuração de credenciais e ID de planilha).

Nota: Este protótipo não implementa todas as regras de negócio da especificação,
mas fornece uma base que pode ser expandida. Ele deve ser executado em um
ambiente onde as bibliotecas `streamlit`, `pandas`, `numpy`, `matplotlib` e
`gspread` estejam instaladas.
"""

import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class CategoryRule:
    """Representa uma regra de categorização baseada em palavras‑chave."""
    category: str
    subcategory: Optional[str]
    keywords: List[str]
    is_fixed: bool = False
    is_extraordinary: bool = False
    is_investment: bool = False
    is_valonni: bool = False


def load_category_rules() -> List[CategoryRule]:
    """
    Carrega as regras de categorização.

    Nesta versão de protótipo, as regras são definidas estaticamente em
    código. Em uma implementação real, você pode carregar de um arquivo JSON
    ou permitir que o usuário edite essas regras dinamicamente.
    """
    rules = [
        CategoryRule(
            category="Transferência interna",
            subcategory=None,
            keywords=["Isabela Valonni", "Transferência interna"],
            is_fixed=False,
        ),
        CategoryRule(
            category="Alimentação",
            subcategory="Restaurantes",
            keywords=[
                "99 Tecnologia",
                "Outback",
                "Clark Patisserie",
                "Passione Per Gelato",
                "Yokubo Restaurante",
            ],
            is_fixed=False,
        ),
        CategoryRule(
            category="Alimentação",
            subcategory="Mercado",
            keywords=["Armazem Urbano", "Supermercados", "Super Mercado", "Vianense"],
            is_fixed=True,
        ),
        CategoryRule(
            category="Carro",
            subcategory="Compra/Manutenção",
            keywords=["Ludi Auto Pecas", "Yasmin", "Diego"],
            is_extraordinary=True,
        ),
        CategoryRule(
            category="Saúde",
            subcategory="Plano de saúde",
            keywords=["Bradesco Saude", "Bradesco Saúde"],
            is_fixed=True,
        ),
        CategoryRule(
            category="Saúde",
            subcategory="Farmácia",
            keywords=["Venancio", "Drogarias"],
            is_fixed=False,
        ),
        CategoryRule(
            category="Moradia",
            subcategory="Aluguel/Condomínio",
            keywords=["Paula", "Carla"],
            is_fixed=True,
        ),
        CategoryRule(
            category="Pets",
            subcategory="Gastos com Pets",
            keywords=["Petshop", "Pets"],
            is_fixed=True,
        ),
        CategoryRule(
            category="Empresa Valonni",
            subcategory="Repasse",  # despesas da Valonni
            keywords=["Leonardo Aires", "Paula C R", "Ande Da Silva Lopes", "Wise"],
            is_valonni=True,
        ),
        CategoryRule(
            category="Investimentos",
            subcategory="Yasmin",
            keywords=["Yasmin"],
            is_investment=True,
        ),
    ]
    return rules


def parse_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """
    Lê o arquivo de extrato e retorna um DataFrame com colunas normalizadas.

    O parser tenta detectar o tipo de arquivo com base na extensão.
    Atualmente suporta CSV e Excel. OFX e PDF devem ser convertidos
    previamente.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Formato de arquivo não suportado. Use CSV ou Excel.")
        return pd.DataFrame()

    # Normalizar nomes de colunas (lowercase, remover espaços)
    df.columns = [c.strip().lower() for c in df.columns]

    # Renomear colunas comuns
    rename_map = {
        "date": "date",
        "data": "date",
        "title": "title",
        "description": "description",
        "amount": "amount",
        "valor": "amount",
        "currency": "currency",
        "paymentmethod": "paymentmethod",
        "status": "status",
        "operationid": "operation_id",
        "activity_id": "activity_id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Converter data para datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        st.warning("Coluna de data não encontrada; tentando usar 'data'.")
        df["date"] = pd.to_datetime(df.get("data", pd.NaT), errors="coerce")

    # Converter valores para float (positivos para entradas, negativos para saídas)
    if "amount" in df.columns:
        # Remover separadores de milhar e trocar vírgula por ponto se necessário
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        st.error("Coluna 'amount' não encontrada.")
        return pd.DataFrame()

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove transações duplicadas com base em operation_id ou activity_id."""
    if "operation_id" in df.columns:
        df = df.drop_duplicates(subset=["operation_id"], keep="first")
    elif "activity_id" in df.columns:
        df = df.drop_duplicates(subset=["activity_id"], keep="first")
    else:
        # Sem identificador único, deduplicar por data, descrição e valor
        df = df.drop_duplicates(subset=["date", "description", "amount"], keep="first")
    return df.reset_index(drop=True)


def categorize_transactions(
    df: pd.DataFrame, rules: List[CategoryRule]
) -> pd.DataFrame:
    """
    Classifica cada transação em uma categoria de acordo com as regras.

    Atribui categoria e subcategoria baseado em palavras‑chave. Se não encontrar
    correspondência, marca como 'Revisar'.
    """
    categories = []
    subcategories = []
    is_fixed_flags = []
    is_extra_flags = []
    is_invest_flags = []
    is_valonni_flags = []

    for _, row in df.iterrows():
        desc = str(row.get("title") or row.get("description") or "").lower()
        matched = False
        for rule in rules:
            if any(keyword.lower() in desc for keyword in rule.keywords):
                categories.append(rule.category)
                subcategories.append(rule.subcategory or "")
                is_fixed_flags.append(rule.is_fixed)
                is_extra_flags.append(rule.is_extraordinary)
                is_invest_flags.append(rule.is_investment)
                is_valonni_flags.append(rule.is_valonni)
                matched = True
                break
        if not matched:
            categories.append("Revisar")
            subcategories.append("")
            is_fixed_flags.append(False)
            is_extra_flags.append(False)
            is_invest_flags.append(False)
            is_valonni_flags.append(False)

    df = df.copy()
    df["category"] = categories
    df["subcategory"] = subcategories
    df["is_fixed"] = is_fixed_flags
    df["is_extraordinary"] = is_extra_flags
    df["is_investment"] = is_invest_flags
    df["is_valonni"] = is_valonni_flags
    return df


def assign_competence(df: pd.DataFrame, cycle_start_day: int = 28) -> pd.DataFrame:
    """
    Atribui o campo 'competence' a cada transação com base no ciclo financeiro.

    O ciclo inicia no dia `cycle_start_day` e termina no dia anterior ao mesmo
    dia do mês seguinte. Por exemplo, se cycle_start_day=28, então
    28/11/2025→27/12/2025 é competência de Dezembro.
    """
    competence = []
    for date in df["date"]:
        if pd.isna(date):
            competence.append(None)
            continue
        day = date.day
        month = date.month
        year = date.year
        if day >= cycle_start_day:
            # Competência é o mês seguinte
            comp_date = date + pd.DateOffset(months=1)
            competence.append(datetime.date(comp_date.year, comp_date.month, 1))
        else:
            competence.append(datetime.date(year, month, 1))
    df = df.copy()
    df["competence"] = competence
    return df


def detect_internal_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta possíveis transferências internas e marca a categoria como
    'Transferência interna'.

    Critério simples: se a descrição contém o nome do titular (Isabela
    Valonni), considera como transferência interna.
    """
    mask = df["category"] == "Revisar"
    df.loc[mask & df["description"].str.contains("isabela valonni", case=False, na=False), "category"] = "Transferência interna"
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa um resumo das transações por competência e categoria.

    Retorna um DataFrame com colunas:
    - competence
    - category
    - total_amount
    - is_fixed
    - is_extraordinary
    - is_investment
    - is_valonni
    """
    if df.empty:
        return pd.DataFrame()

    df_grouped = df.groupby([
        "competence",
        "category",
        "is_fixed",
        "is_extraordinary",
        "is_investment",
        "is_valonni",
    ], as_index=False)["amount"].sum()
    df_grouped = df_grouped.rename(columns={"amount": "total_amount"})
    return df_grouped


def summary_by_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separa o resumo em despesas pessoais, despesas Valonni e investimentos.

    Retorna três DataFrames: (pessoal, valonni, investimentos)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pessoal = df[~df["is_valonni"] & ~df["is_investment"]]
    valonni = df[df["is_valonni"]]
    invest = df[df["is_investment"]]
    return pessoal, valonni, invest


def display_dashboard():
    """Renderiza a interface principal do aplicativo Streamlit."""
    st.set_page_config(page_title="Controle Financeiro Pessoal", layout="wide")
    st.title("Controle Financeiro Pessoal")

    st.sidebar.header("Parâmetros")
    cycle_start_day = st.sidebar.number_input(
        "Dia de início do ciclo (competência)", value=28, min_value=1, max_value=31
    )

    # Upload de arquivo
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do extrato (CSV ou Excel)", type=["csv", "xls", "xlsx"]
    )

    data_df = pd.DataFrame()
    if uploaded_file is not None:
        with st.spinner("Carregando dados..."):
            raw_df = parse_file(uploaded_file)
            raw_df = remove_duplicates(raw_df)
            rules = load_category_rules()
            normalized_df = categorize_transactions(raw_df, rules)
            normalized_df = detect_internal_transfers(normalized_df)
            normalized_df = assign_competence(normalized_df, cycle_start_day)
            data_df = normalized_df

    if not data_df.empty:
        st.subheader("Visualização das transações")
        st.dataframe(data_df.head(1000))

        st.subheader("Resumo por competência e categoria")
        summary_df = compute_summary(data_df)
        pessoal_df, valonni_df, invest_df = summary_by_type(summary_df)

        st.write("### Despesas pessoais (exclui Valonni e investimentos)")
        if not pessoal_df.empty:
            st.dataframe(pessoal_df)
        else:
            st.info("Nenhuma transação pessoal encontrada.")

        st.write("### Despesas Valonni (informativas)")
        if not valonni_df.empty:
            st.dataframe(valonni_df)
        else:
            st.info("Nenhuma transação da empresa Valonni encontrada.")

        st.write("### Investimentos e aplicações")
        if not invest_df.empty:
            st.dataframe(invest_df)
        else:
            st.info("Nenhuma transação de investimento encontrada.")

        # Gráfico simples de despesas por categoria
        st.subheader("Gráfico de despesas por categoria (pessoal)")
        import matplotlib.pyplot as plt

        # Agrupar despesas pessoais por categoria
        cat_summary = (
            pessoal_df.groupby("category", as_index=False)["total_amount"].sum().sort_values(by="total_amount", ascending=False)
        )
        if not cat_summary.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(cat_summary["category"], cat_summary["total_amount"])
            ax.set_xlabel("Valor (R$")")
            ax.set_ylabel("Categoria")
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.info("Nenhuma despesa pessoal para exibir no gráfico.")

        st.write("\n")

    else:
        st.info("Faça upload de um extrato para começar.")


if __name__ == "__main__":
    display_dashboard()
