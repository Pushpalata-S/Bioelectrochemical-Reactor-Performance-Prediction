"""
Bioelectrochemical Reactor Performance Prediction Dashboard
Main Streamlit Application Entry Point & Navigation Router.
"""
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Bioelectrochemical Reactor Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import CUSTOM_CSS
from src.pages.overview import render_overview_page
from src.pages.electricity import render_electricity_page
from src.pages.wastewater import render_wastewater_page
from src.pages.evaluation import render_evaluation_page
from src.pages.explorer import render_explorer_page
from src.pages.scenarios import render_scenarios_page

# Inject custom CSS styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    # Sidebar Navigation & Platform Header
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.8rem;">⚡</span>
                <div>
                    <h3 style="margin: 0; color: #FFFFFF; font-size: 1.15rem; font-weight: 700; line-height: 1.2;">
                        MFC Analytics
                    </h3>
                    <span style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">
                        Bioelectrochemical Systems
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Page Navigation
        nav_options = [
            "🏛️ Project Overview",
            "⚡ Electricity Generation",
            "💧 Wastewater Treatment",
            "📈 Model Benchmarking",
            "🔬 Dataset Explorer",
            "🧪 Scenario Comparison",
        ]

        selected_page = st.radio(
            "Navigation Menu:",
            nav_options,
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Sidebar Information Cards
        st.markdown("""
        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 0.85rem; margin-bottom: 1rem;">
            <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">
                Platform Status
            </div>
            <div style="margin-top: 0.4rem; font-size: 0.85rem; color: #E2E8F0;">
                • <strong>Records:</strong> 67 experimental rows<br>
                • <strong>Validation:</strong> Study-Grouped 5-Fold<br>
                • <strong>Mode 1 Model:</strong> Random Forest<br>
                • <strong>Mode 2 Model:</strong> Gaussian Process
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size: 0.75rem; color: #64748B; text-align: center; margin-top: 1rem;">
            Chemical Engineering Capstone<br>
            <a href="https://github.com/Pushpalata-S/Bioelectrochemical-Reactor-Performance-Prediction" target="_blank" style="color: #14B8A6; text-decoration: none;">
                GitHub Repository ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

    # Main Body Router
    if selected_page == "🏛️ Project Overview":
        render_overview_page()
    elif selected_page == "⚡ Electricity Generation":
        render_electricity_page()
    elif selected_page == "💧 Wastewater Treatment":
        render_wastewater_page()
    elif selected_page == "📈 Model Benchmarking":
        render_evaluation_page()
    elif selected_page == "🔬 Dataset Explorer":
        render_explorer_page()
    elif selected_page == "🧪 Scenario Comparison":
        render_scenarios_page()


if __name__ == "__main__":
    main()
