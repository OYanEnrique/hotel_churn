import os

import requests
import streamlit as st


st.set_page_config(page_title="Hotel Churn", page_icon="🏨", layout="centered")

st.title("Hotel Churn Predictor")
st.caption("Uma interface guiada para prever churn com a API do projeto.")

st.write(
    "Preencha o perfil do hóspede abaixo. O app envia os dados para a API local e retorna a previsão."
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/prever_churn")

FIELD_LABELS = {
    "lead_time": "Com quantos dias de antecedência a reserva foi feita?",
    "adr": "Qual foi a tarifa média diária da reserva?",
    "total_pessoas": "Quantas pessoas estão incluídas na reserva?",
    "total_noites": "Quantas noites a estadia terá?",
    "mudou_quarto": "O cliente mudou de quarto durante a estadia?",
    "previous_cancellations": "Quantos cancelamentos esse cliente já fez no passado?",
    "hotel": "Qual tipo de hotel?",
    "market_segment": "Qual foi o canal ou segmento de mercado da reserva?",
    "deposit_type": "Qual tipo de depósito foi escolhido?",
    "customer_type": "Qual é o perfil do cliente?",
    "tem_filhos": "A reserva inclui crianças ou bebês?",
}

with st.form("form_churn"):
    st.subheader("Detalhes da reserva")
    col1, col2 = st.columns(2)

    with col1:
        lead_time = st.number_input(FIELD_LABELS["lead_time"], min_value=0, value=30, step=1)
        adr = st.number_input(FIELD_LABELS["adr"], min_value=0.0, value=100.0, step=1.0)
        total_pessoas = st.number_input(FIELD_LABELS["total_pessoas"], min_value=1, value=2, step=1)
        total_noites = st.number_input(FIELD_LABELS["total_noites"], min_value=1, value=2, step=1)
        mudou_quarto = st.selectbox(FIELD_LABELS["mudou_quarto"], options=[0, 1], index=0, format_func=lambda x: "Não" if x == 0 else "Sim")

    with col2:
        previous_cancellations = st.number_input(
            FIELD_LABELS["previous_cancellations"], min_value=0, value=0, step=1
        )
        hotel = st.selectbox(FIELD_LABELS["hotel"], options=["Resort Hotel", "City Hotel"], index=0)
        market_segment = st.selectbox(
            FIELD_LABELS["market_segment"],
            options=["Direct", "Corporate", "Online TA", "Offline TA/TO", "Groups", "Complementary"],
            index=0,
        )
        deposit_type = st.selectbox(
            FIELD_LABELS["deposit_type"],
            options=["No Deposit", "Non Refund", "Refundable"],
            index=0,
        )
        customer_type = st.selectbox(
            FIELD_LABELS["customer_type"],
            options=["Transient", "Transient-Party", "Contract", "Group"],
            index=0,
        )

    tem_filhos = st.selectbox(
        FIELD_LABELS["tem_filhos"],
        options=["no", "yes"],
        index=0,
        format_func=lambda x: "Não" if x == "no" else "Sim",
    )

    submitted = st.form_submit_button("Analisar risco de churn")

if submitted:
    payload = {
        "lead_time": int(lead_time),
        "adr": float(adr),
        "total_pessoas": int(total_pessoas),
        "total_noites": int(total_noites),
        "mudou_quarto": int(mudou_quarto),
        "previous_cancellations": int(previous_cancellations),
        "hotel": "Resort" if hotel == "Resort Hotel" else "City",
        "market_segment": market_segment,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "tem_filhos": tem_filhos,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            churn = response.json().get("churn_predito")
            if churn == 1:
                st.error("Previsão: este hóspede tem maior risco de churn.")
            else:
                st.success("Previsão: este hóspede tem menor risco de churn.")
            st.json(response.json())
        else:
            st.error(f"Erro da API ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Não foi possível acessar a API em {API_URL}: {exc}")
