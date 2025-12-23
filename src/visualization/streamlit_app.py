"""Streamlit dashboard for dish similarity visualization.

Connects to Neo4j to fetch dish data and displays pairwise similarity
matrices using three different metrics: Jaccard, ingredient embeddings,
and image embeddings.

Run with: streamlit run src/visualization/streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Add project root to path to enable absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from urllib.parse import quote
from difflib import SequenceMatcher
from matplotlib.figure import Figure

from src.services.neo4j_service import Neo4jService
from src.visualization.dish_aggregator import get_aggregator
from src.visualization.similarity import (
    compute_jaccard_matrix,
    compute_ingredient_embedding_matrix,
    compute_image_embedding_matrix,
)
from src.visualization.recommendation_page import render_recommendations_page
from src.visualization.upload_page import render_upload_page


# Page configuration
st.set_page_config(
    page_title="Dish Similarity Matrix",
    page_icon="🍽️",
    layout="wide",
)


@st.cache_data(ttl=60)
def api_get_pending_ingredients(api_base_url: str) -> list[dict]:
    """Fetch pending ingredients via FastAPI."""
    resp = requests.get(f"{api_base_url.rstrip('/')}/ingredients/pending", timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("pending", [])


@st.cache_data(ttl=300)
def api_get_all_ingredients(api_base_url: str) -> list[str]:
    """Fetch all ingredients via FastAPI (used for merge lookup)."""
    resp = requests.get(f"{api_base_url.rstrip('/')}/ingredients", timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("ingredients", [])


@st.cache_data(ttl=300)
def api_get_dietary_restrictions(api_base_url: str) -> list[dict]:
    """Fetch dietary restrictions via FastAPI."""
    resp = requests.get(f"{api_base_url.rstrip('/')}/dietary-restrictions", timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("restrictions", [])


def api_approve_ingredient(
    api_base_url: str,
    name: str,
    suited_for: list[str] | None = None,
    not_suited_for: list[str] | None = None,
) -> dict:
    resp = requests.post(
        f"{api_base_url.rstrip('/')}/ingredients/{quote(name)}/approve",
        json={
            "suited_for": suited_for or [],
            "not_suited_for": not_suited_for or [],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def api_reject_and_merge_ingredient(api_base_url: str, name: str, merge_into: str) -> dict:
    resp = requests.post(
        f"{api_base_url.rstrip('/')}/ingredients/{quote(name)}/reject",
        json={"merge_into": merge_into},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def render_pending_ingredients_page(api_base_url: str) -> None:
    st.title("🧪 Pending Ingredients")
    st.markdown("Review pending ingredients and either approve them or merge them into an existing ingredient.")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        refresh = st.button("Refresh", help="Reload pending and ingredient lists")
    with col_b:
        if st.button("Clear Cache"):
            api_get_pending_ingredients.clear()
            api_get_all_ingredients.clear()
            st.rerun()

    if refresh:
        api_get_pending_ingredients.clear()
        api_get_all_ingredients.clear()
        st.rerun()

    if not api_base_url:
        st.error("Please enter a FastAPI base URL.")
        return

    # Load data via API
    try:
        pending = api_get_pending_ingredients(api_base_url)
    except Exception as e:
        st.error(f"Failed to fetch pending ingredients: {e}")
        return

    if not pending:
        st.info("No pending ingredients.")
        return

    try:
        all_ingredients = api_get_all_ingredients(api_base_url)
    except Exception as e:
        st.error(f"Failed to fetch ingredient list for merge lookup: {e}")
        return

    try:
        restrictions_payload = api_get_dietary_restrictions(api_base_url)
        restriction_names = [r.get("name") for r in restrictions_payload if r.get("name")]
    except Exception as e:
        st.error(f"Failed to fetch dietary restrictions: {e}")
        return

    st.sidebar.markdown(f"**Pending:** {len(pending)}")
    st.sidebar.markdown(f"**All ingredients:** {len(all_ingredients)}")
    st.sidebar.markdown(f"**Dietary restrictions:** {len(restriction_names)}")

    # Selection
    pending_names = [p.get("name", "") for p in pending if p.get("name")]
    selected = str(st.selectbox("Select a pending ingredient", options=pending_names))
    selected_meta = next((p for p in pending if p.get("name") == selected), {})

    st.write(
        {
            "name": selected,
            "dish_count": selected_meta.get("dish_count", 0),
            "created_at": selected_meta.get("created_at"),
            "pending_confidence": selected_meta.get("pending_confidence"),
            "pending_best_score": selected_meta.get("pending_best_score"),
            "suggested_merges": selected_meta.get("suggested_merges", []),
        }
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Approve")
        st.caption("Marks the ingredient as canonical and optionally links dietary restrictions.")

        suited_for = st.multiselect(
            "SUITED_FOR restrictions",
            options=restriction_names,
            default=[],
            help="Select restrictions this ingredient is suitable for (creates SUITED_FOR relationships)",
        )
        not_suited_for = st.multiselect(
            "NOT_SUITED_FOR restrictions",
            options=restriction_names,
            default=[],
            help="Select restrictions this ingredient is NOT suitable for (creates NOT_SUITED_FOR relationships)",
        )

        if st.button("Approve ingredient", type="primary"):
            try:
                api_approve_ingredient(api_base_url, selected, suited_for, not_suited_for)
                api_get_pending_ingredients.clear()
                api_get_all_ingredients.clear()
                api_get_dietary_restrictions.clear()
                st.success(f"Approved '{selected}'")
                st.rerun()
            except Exception as e:
                st.error(f"Approve failed: {e}")

    with col2:
        st.subheader("Merge")
        st.caption("Merges dishes from this ingredient into an existing ingredient and deletes the pending one.")

        # Prevent selecting itself as target
        merge_options = [x for x in all_ingredients if x and x != selected]

        # Suggested target: prefer stored suggestions, otherwise fuzzy match by name
        suggested_from_model = None
        model_suggestions = selected_meta.get("suggested_merges", []) or []
        if model_suggestions and isinstance(model_suggestions, list):
            first = model_suggestions[0]
            if isinstance(first, dict) and first.get("name"):
                suggested_from_model = str(first.get("name"))

        suggested_fuzzy = None
        fuzzy_score = 0.0
        if merge_options:
            best = (None, 0.0)
            for candidate in merge_options:
                score = SequenceMatcher(None, selected.lower(), str(candidate).lower()).ratio()
                if score > best[1]:
                    best = (candidate, score)
            suggested_fuzzy, fuzzy_score = best

        suggested_target = suggested_from_model or suggested_fuzzy

        if suggested_target:
            if suggested_from_model:
                st.caption(f"Suggested merge (from candidates): {suggested_target}")
            else:
                st.caption(f"Suggested merge (fuzzy): {suggested_target} (ratio={fuzzy_score:.2f})")

        default_index = 0
        if suggested_target and suggested_target in merge_options:
            default_index = merge_options.index(suggested_target)

        merge_into = str(st.selectbox("Merge into", options=merge_options, index=default_index))

        if st.button("Merge and delete pending", type="secondary"):
            try:
                api_reject_and_merge_ingredient(api_base_url, selected, merge_into)
                api_get_pending_ingredients.clear()
                api_get_all_ingredients.clear()
                st.success(f"Merged '{selected}' → '{merge_into}'")
                st.rerun()
            except Exception as e:
                st.error(f"Merge failed: {e}")


@st.cache_resource
def get_neo4j_service() -> Neo4jService:
    """Get cached Neo4j service instance."""
    service = Neo4jService()
    service.verify_connectivity()
    return service


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_dishes_summary(_service: Neo4jService, limit: int) -> list[dict]:
    """Fetch dish summary from Neo4j with caching."""
    return _service.get_dishes_summary(limit=limit)


@st.cache_data(ttl=300)
def fetch_random_dishes_summary(_service: Neo4jService, limit: int) -> list[dict]:
    """Fetch random dish summary from Neo4j with caching."""
    return _service.get_random_dishes_summary(limit=limit)


@st.cache_data(ttl=60)
def search_dishes_summary(_service: Neo4jService, query: str, limit: int) -> list[dict]:
    """Search dishes by name from Neo4j with caching."""
    return _service.search_dishes_summary(query, limit=limit)


@st.cache_resource
def fit_aggregator_idf(_service: Neo4jService) -> None:
    """Fit the TF-IDF aggregator with all recipes from the database.
    
    This should be called once on app startup to compute IDF weights
    for all ingredients across all dishes.
    """
    # Fetch all dishes with their ingredients
    all_dishes = _service.get_all_dishes_ingredients()
    
    if not all_dishes:
        st.warning("No dishes found for IDF fitting. Using default weights.")
        return
    
    # Extract ingredient lists for each dish
    all_recipes = [dish.get("ingredients", []) for dish in all_dishes]
    
    # Fit the aggregator
    aggregator = get_aggregator(method='tfidf')
        # 1. Fetch all recipes (dish ingredients) for IDF calculation
    # (You likely already have this part)
    dishes_data = _service.get_all_dishes_ingredients()
    all_recipes = [d['ingredients'] for d in dishes_data]

    # 2. Fetch all unique ingredient embeddings for Global Mean Centering
    print("Fetching ingredient embeddings from Neo4j...")
    _, all_embeddings = _service.get_all_ingredient_embeddings()
    
    # 3. Fit the aggregator with both datasets
    # This calculates IDF *and* the Global Mean vector
    aggregator.fit_idf(
        all_recipes=all_recipes, 
        all_ingredient_embeddings=all_embeddings
    )
    weights = list(aggregator.idf_map.values())
    print(f"Min IDF: {min(weights)}, Max IDF: {max(weights)}")

@st.cache_data(ttl=300)
def fetch_dishes_with_embeddings(
    _service: Neo4jService,
    dish_ids: tuple[str, ...],
) -> list[dict]:
    """Fetch dishes with embeddings from Neo4j with caching."""
    return _service.get_dishes_with_embeddings_and_ingredients(list(dish_ids))


def render_plotly_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
) -> go.Figure:
    """Render an interactive Plotly heatmap."""
    fig = px.imshow(
        matrix,
        x=labels,
        y=labels,
        color_continuous_scale="RdYlGn",
        aspect="equal",
        zmin=0,
        zmax=1,
        labels={"color": "Similarity"},
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(tickangle=45, side="bottom"),
        yaxis=dict(tickangle=0),
        width=800,
        height=800,
    )
    return fig


def render_seaborn_clustermap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
) -> Figure:
    """Render a hierarchical clustermap using seaborn."""
    # Create clustermap with hierarchical clustering
    g = sns.clustermap(
        matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        figsize=(12, 12),
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        method="average",  # Linkage method for hierarchical clustering
    )
    g.fig.suptitle(title, y=1.02, fontsize=14)
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=8,
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        rotation=0,
        fontsize=8,
    )
    return g.fig


def main():
    """Main Streamlit application."""
    # Sidebar navigation
    st.sidebar.header("⚙️ Settings")
    
    # API Configuration
    st.sidebar.subheader("API")
    default_api_url = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
    api_base_url = st.sidebar.text_input(
        "FastAPI base URL",
        value=default_api_url,
        help="Example: http://localhost:8000/api/v1",
    ).strip()

    page = st.sidebar.radio(
        "Page",
        options=["Dish Similarity", "Upload & Process", "Pending Ingredients", "Dish Recommendations"],
        index=0,
    )

    if page == "Upload & Process":
        render_upload_page(api_base_url)
        return
    elif page == "Pending Ingredients":
        render_pending_ingredients_page(api_base_url)
        return
    elif page == "Dish Recommendations":
        render_recommendations_page(api_base_url)
        return

    st.title("🍽️ Dish Similarity Matrix")
    st.markdown("Visualize pairwise similarity between dishes using different metrics.")

    # Initialize Neo4j connection
    try:
        neo4j_service = get_neo4j_service()
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        st.info("Make sure Neo4j is running and check your connection settings.")
        return

    # Fit TF-IDF aggregator with all recipes (cached, runs once)
    with st.spinner("Fitting TF-IDF weights..."):
        fit_aggregator_idf(neo4j_service)

    # Refresh button
    if st.sidebar.button("Refresh Data", help="Reload dishes and recompute IDF weights"):
        with st.spinner("Refreshing data..."):
            fetch_dishes_summary.clear()
            fetch_dishes_with_embeddings.clear()
            fit_aggregator_idf.clear()
            fit_aggregator_idf(neo4j_service)
        st.sidebar.success("Data refreshed")
        st.rerun()

    # Selection Mode
    selection_mode = st.sidebar.radio(
        "Selection Mode",
        options=["Top N Dishes", "Random N Dishes", "Manual Search"],
        index=0,
    )

    dishes_summary = []

    if selection_mode == "Top N Dishes":
        # Max dishes limit
        max_dishes = st.sidebar.number_input(
            "Maximum dishes to load",
            min_value=2,
            max_value=100,
            value=30,
            step=5,
            help="Limit the number of dishes loaded from the database",
        )
        # Fetch available dishes
        with st.spinner("Loading dishes from database..."):
            dishes_summary = fetch_dishes_summary(neo4j_service, limit=max_dishes)

    elif selection_mode == "Random N Dishes":
        max_dishes = st.sidebar.number_input(
            "Number of random dishes",
            min_value=2,
            max_value=100,
            value=30,
            step=5,
        )
        if st.sidebar.button("Shuffle Random Dishes"):
             fetch_random_dishes_summary.clear()
             
        with st.spinner("Loading random dishes..."):
            dishes_summary = fetch_random_dishes_summary(neo4j_service, limit=max_dishes)

    elif selection_mode == "Manual Search":
        st.sidebar.markdown("### Search & Add")
        search_query = st.sidebar.text_input("Search dish by name")
        
        # Initialize session state for manually added dishes if not exists
        if "manual_dishes" not in st.session_state:
            st.session_state.manual_dishes = []
            
        if search_query:
            results = search_dishes_summary(neo4j_service, search_query, limit=10)
            if results:
                # Create a selectbox for results
                options = {d["name"]: d for d in results}
                selected_to_add = st.sidebar.selectbox("Found dishes", options=list(options.keys()))
                
                if st.sidebar.button("Add to Selection"):
                    dish_to_add = options[selected_to_add]
                    # Check if already added
                    if not any(d["dish_id"] == dish_to_add["dish_id"] for d in st.session_state.manual_dishes):
                        st.session_state.manual_dishes.append(dish_to_add)
                        st.sidebar.success(f"Added {selected_to_add}")
                    else:
                        st.sidebar.warning("Dish already in selection")
            else:
                st.sidebar.info("No dishes found")
        
        if st.sidebar.button("Clear Selection"):
            st.session_state.manual_dishes = []
            st.rerun()
            
        dishes_summary = st.session_state.manual_dishes
        
        if not dishes_summary:
             st.info("Search and add dishes to compare.")

    if not dishes_summary:
        if selection_mode == "Manual Search":
            return
        st.warning("No dishes found with image embeddings in the database.")
        st.info("Please register some dishes first using the API.")
        return

    # Create dish name to ID mapping
    dish_options = {d["name"]: d["dish_id"] for d in dishes_summary}
    dish_names = list(dish_options.keys())

    st.sidebar.markdown(f"**Found {len(dish_names)} dishes**")

    # Multi-select for dishes
    selected_names = st.sidebar.multiselect(
        "Select dishes to compare",
        options=dish_names,
        default=dish_names,
        help="Choose dishes to include in the similarity matrix",
    )

    if len(selected_names) < 2:
        st.warning("Please select at least 2 dishes to compare.")
        return

    # Metric selection
    metric = st.sidebar.radio(
        "Similarity Metric",
        options=["Jaccard (Ingredients)", "Ingredient Embeddings", "Image Embeddings"],
        index=0,
        help=(
            "**Jaccard**: Based on shared ingredients (set overlap)\n\n"
            "**Ingredient Embeddings**: Cosine similarity of averaged ingredient vectors\n\n"
            "**Image Embeddings**: Cosine similarity of dish images (visual similarity)"
        ),
    )

    # Visualization type toggle
    viz_type = st.sidebar.radio(
        "Visualization Type",
        options=["Heatmap (Plotly)", "Clustermap (Seaborn)"],
        index=0,
        help=(
            "**Heatmap**: Interactive, zoomable visualization\n\n"
            "**Clustermap**: Includes hierarchical clustering dendrograms"
        ),
    )

    # Fetch full dish data
    selected_ids = tuple(dish_options[name] for name in selected_names)

    with st.spinner("Fetching dish embeddings..."):
        dishes = fetch_dishes_with_embeddings(neo4j_service, selected_ids)

    if not dishes:
        st.error("Failed to fetch dish data.")
        return

    # Compute similarity matrix based on selected metric
    with st.spinner("Computing similarity matrix..."):
        if metric == "Jaccard (Ingredients)":
            matrix, labels = compute_jaccard_matrix(dishes)
            skipped = []
        elif metric == "Ingredient Embeddings":
            matrix, labels, skipped = compute_ingredient_embedding_matrix(dishes)
        else:  # Image Embeddings
            matrix, labels, skipped = compute_image_embedding_matrix(dishes)

    # Show warnings for skipped dishes
    if skipped:
        st.warning(
            f"⚠️ {len(skipped)} dish(es) skipped due to missing embeddings: "
            f"{', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}"
        )

    if len(labels) < 2:
        st.error("Not enough dishes with valid data to compute similarity matrix.")
        return

    # Display matrix statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dishes Compared", len(labels))
    with col2:
        # Get upper triangle values (excluding diagonal)
        upper_tri = matrix[np.triu_indices(len(labels), k=1)]
        st.metric("Mean Similarity", f"{np.mean(upper_tri):.3f}")
    with col3:
        st.metric("Min Similarity", f"{np.min(upper_tri):.3f}")
    with col4:
        st.metric("Max Similarity", f"{np.max(upper_tri):.3f}")

    # Render visualization
    st.markdown("---")

    if viz_type == "Heatmap (Plotly)":
        title = f"Dish Similarity Matrix ({metric})"
        fig = render_plotly_heatmap(matrix, labels, title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        title = f"Dish Similarity Clustermap ({metric})"
        fig = render_seaborn_clustermap(matrix, labels, title)
        st.pyplot(fig)
        plt.close(fig)

    # Show raw data in expander
    with st.expander("📊 View Raw Similarity Data"):
        import pandas as pd

        df = pd.DataFrame(matrix, index=labels, columns=labels)
        st.dataframe(df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1))

    # Most similar pairs
    with st.expander("🔗 Most Similar Pairs"):
        pairs = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs.append({
                    "Dish A": labels[i],
                    "Dish B": labels[j],
                    "Similarity": matrix[i, j],
                })
        import pandas as pd

        pairs_df = pd.DataFrame(pairs)
        pairs_df = pairs_df.sort_values("Similarity", ascending=False).head(20)
        st.dataframe(pairs_df, use_container_width=True)


if __name__ == "__main__":
    main()
