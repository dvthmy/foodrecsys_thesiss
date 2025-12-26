"""Streamlit page for dish recommendations.

Displays recommendations using both Collaborative Filtering and Content-Based Filtering.
"""

import streamlit as st
import pandas as pd
import requests
from typing import Any


@st.cache_data(ttl=300)
def api_get_dietary_restrictions(api_base_url: str) -> list[dict]:
    """Fetch available dietary restrictions (allergies/diets)."""
    try:
        resp = requests.get(f"{api_base_url.rstrip('/')}/dietary-restrictions", timeout=20)
        resp.raise_for_status()
        return resp.json().get("restrictions", [])
    except Exception as e:
        st.error(f"Failed to fetch dietary restrictions: {e}")
        return []


def api_get_users(api_base_url: str) -> list[dict]:
    """Fetch all users."""
    try:
        resp = requests.get(f"{api_base_url.rstrip('/')}/users", params={"limit": 1000}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("users", [])
    except Exception as e:
        st.error(f"Failed to fetch users: {e}")
        return []


def api_create_user(
    api_base_url: str,
    name: str,
    age: int | None = None,
    gender: str | None = None,
    nationality: str | None = None,
    dietary_restrictions: list[str] | None = None,
) -> dict | None:
    """Create a new user."""
    try:
        resp = requests.post(
            f"{api_base_url.rstrip('/')}/users",
            json={
                "name": name,
                "age": age,
                "gender": gender,
                "nationality": nationality,
                "dietary_restrictions": dietary_restrictions or [],
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to create user: {e}")
        return None


def api_get_recommendations(api_base_url: str, user_id: str, method: str, metric: str) -> dict:
    """Fetch recommendations for a user."""
    try:
        resp = requests.get(
            f"{api_base_url.rstrip('/')}/users/{user_id}/recommendations",
            params={"limit": 3, "method": method, "metric": metric},
            timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch recommendations ({method}, {metric}): {e}")
        return {}

def api_get_similar_users(api_base_url: str, user_id: str) -> dict:
    """Fetch similar users."""
    try:
        resp = requests.get(
            f"{api_base_url.rstrip('/')}/users/{user_id}/similar",
            params={"limit": 5},
            timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # Don't show error for similar users as it might be empty/irrelevant for content-based
        return {}


def api_get_excluded_dishes(api_base_url: str, user_id: str) -> dict:
    """Fetch dishes excluded by user's dietary restrictions."""
    try:
        resp = requests.get(
            f"{api_base_url.rstrip('/')}/users/{user_id}/excluded-dishes",
            params={"limit": 3},
            timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {}

def api_search_by_image(api_base_url: str, image_file: Any) -> dict:
    """Search for dishes by image."""
    try:
        files = {"image": (image_file.name, image_file, image_file.type)}
        resp = requests.post(
            f"{api_base_url.rstrip('/')}/dishes/search-by-image",
            files=files,
            params={"limit": 5, "threshold": 0.6},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
        return {}

def render_recommendation_card(dish: dict, method: str):
    """Render a single recommendation card."""
    with st.container(border=True):
        st.subheader(dish["name"])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            score_label = "Predicted Rating" if "collaborative" in method else "Score"
            st.metric(score_label, f"{dish['predicted_score']:.2f}")
            
            if "collaborative" in method:
                st.caption(f"Based on {dish['recommender_count']} similar users")
        
        with col2:
            if dish.get("description"):
                st.write(dish["description"])
            
            if dish.get("ingredients"):
                st.caption(f"**Ingredients:** {', '.join(dish['ingredients'][:5])}...")

def render_recommendations_page(api_base_url: str = "https://api.foodrecys.captechvn.com/api/v1") -> None:
    """Render the recommendation page."""
    st.title("🍽️ Dish Recommendations")
    
    tab1, tab2 = st.tabs(["User Recommendations", "Image Search"])
    
    with tab1:
        # User Selection
        users = api_get_users(api_base_url)
        restrictions = api_get_dietary_restrictions(api_base_url)
        restriction_options = ["No dietary filter"] + [r["name"] for r in restrictions] if restrictions else ["No dietary filter"]
        if not users:
            st.warning("No users found. Please ingest users first.")
        
        # Add New User Section (for cold start demonstration)
        with st.expander("Create New User", expanded=False):
            st.info(
                "**Cold Start Scenario:** When a user has < 3 ratings, the system cannot find similar users "
                "for collaborative filtering. In this case, it falls back to **popular dish recommendations**."
            )
            
            col_name, col_age = st.columns(2)
            with col_name:
                new_user_name = st.text_input("Name", key="new_user_name", placeholder="Enter user name")
            with col_age:
                new_user_age = st.number_input("Age (optional)", min_value=1, max_value=120, value=None, key="new_user_age")
            
            col_gender, col_nationality = st.columns(2)
            with col_gender:
                new_user_gender = st.selectbox("Gender (optional)", options=["", "Male", "Female", "Other"], key="new_user_gender")
            with col_nationality:
                new_user_nationality = st.text_input("Nationality (optional)", key="new_user_nationality")
            
            restriction_names = [r["name"] for r in restrictions] if restrictions else []
            new_user_restrictions = st.multiselect(
                "Dietary Restrictions (optional)",
                options=restriction_names,
                key="new_user_restrictions",
            )
            
            if st.button("Create User", type="primary", key="create_user_btn"):
                if not new_user_name or not new_user_name.strip():
                    st.error("Please enter a user name.")
                else:
                    result = api_create_user(
                        api_base_url,
                        name=new_user_name.strip(),
                        age=new_user_age if new_user_age else None,
                        gender=new_user_gender if new_user_gender else None,
                        nationality=new_user_nationality.strip() if new_user_nationality else None,
                        dietary_restrictions=new_user_restrictions if new_user_restrictions else None,
                    )
                    if result and result.get("success"):
                        st.success(f" {result.get('message', 'User created!')}")
                        st.caption(f"User ID: `{result.get('user_id')}`")
                        st.rerun()
        
        if users:
            # Row 1: dietary filter + user
            col_diet, col_user = st.columns([1.5, 2])
            with col_diet:
                selected_restriction = st.selectbox(
                    "Dietary Mode",
                    options=restriction_options,
                    help="Choose allergy/diet filter (informational; applies when backend supports it).",
                )
                diet_filter = None if selected_restriction == "No dietary filter" else selected_restriction
                # Apply dietary filter to users list
                if diet_filter:
                    filtered_users = [
                        u for u in users
                        if diet_filter in (u.get("dietary_restrictions") or [])
                    ]
                else:
                    filtered_users = users

            with col_user:
                
                if filtered_users:
                    # Build user options with short ID
                    all_user_options = {f"{u['name']} ({u['user_id'][:8]}...)": u['user_id'] for u in filtered_users}
                    
                    # Single searchable selectbox
                    selected_user_label = st.selectbox(
                        "Select User",
                        options=[""] + list(all_user_options.keys()),  # Empty first option
                        index=0,  # Start with empty selection
                        placeholder="Type to search user...",
                        key=f"user_select_{diet_filter or 'all'}",
                    )
                    
                    # Handle empty selection
                    if not selected_user_label:
                        selected_user_label = None
                else:
                    st.warning("No users match this dietary mode.")
                    selected_user_label = None
            
            # Row 2: method + metric
            col_method, col_metric = st.columns([1.5, 1.5])
            with col_method:
                method_option = st.selectbox(
                    "Recommendation Method",
                    options=["Content-Based (Ingredients)", "Collaborative Filtering"],
                    help="Content-Based uses ingredient similarity. Collaborative uses similar users. (Image-based recommendations available in Image Search tab)"
                )
            with col_metric:
                # Determine method and metric based on selection
                if "Content-Based" in method_option:
                    method_param = "content_based"
                    metric_option = st.selectbox(
                        "Content-Based Metric",
                        options=["Jaccard (Set Overlap)", "Ingredient Embedding (Semantic)"],
                        help="Jaccard uses ingredient set overlap. Ingredient Embedding uses Gemma semantic embeddings with TF-IDF weighting."
                    )
                    cb_metric = "jaccard" if "Jaccard" in metric_option else "ingredient_embedding"
                    cf_metric = "cosine"  # Not used for content-based
                else:
                    method_param = "collaborative"
                    cf_metric = "cosine"  # Only use Cosine for CF
                    cb_metric = None  # Not used for collaborative
            
            if selected_user_label:
                selected_user_id = all_user_options[selected_user_label]
                
                # Fetch Data
                if st.button("Get Recommendations"):
                    if method_param == "content_based":
                        # Content-Based Recommendations (Full Width)
                        metric_name = "Jaccard (Set Overlap)" if cb_metric == "jaccard" else "Ingredient Embedding (Semantic)"
                        st.header("🥘 Content-Based Recommendations")
                        st.caption(f"Recommends dishes similar to what you like ({metric_name}).")
                        if diet_filter:
                            st.caption(f"Dietary filter: {diet_filter} (if available in backend)")
                        
                        with st.spinner("Fetching content-based recommendations..."):
                            rec_data = api_get_recommendations(api_base_url, selected_user_id, "content_based", cb_metric)
                        
                        if rec_data:
                            recs = rec_data.get("recommendations", [])
                            if not recs:
                                st.info("No recommendations found. Try rating more dishes.")
                            else:
                                for dish in recs:
                                    render_recommendation_card(dish, "content_based")
                    
                    else:
                        # Collaborative Filtering (Full Width)
                        st.header("👥 Collaborative Filtering")
                        st.caption("Recommends dishes liked by similar users (Cosine Similarity).")
                        if diet_filter:
                            st.caption(f"Dietary filter: {diet_filter} (if available in backend)")
                        
                        with st.spinner("Fetching collaborative filtering recommendations..."):
                            rec_data = api_get_recommendations(api_base_url, selected_user_id, "collaborative", cf_metric)
                        
                        if rec_data:
                            method_used = rec_data.get("method")
                            if "popular_fallback" in method_used:
                                st.warning("⚠️ Not enough data for collaborative filtering. Showing popular dishes instead.")
                            
                            recs = rec_data.get("recommendations", [])
                            if not recs:
                                st.info("No recommendations found.")
                            else:
                                for dish in recs:
                                    render_recommendation_card(dish, "collaborative")
                            
                            # Show similar users
                            if "collaborative_filtering" in method_used:
                                with st.expander("👥 Similar Users"):
                                    sim_data = api_get_similar_users(api_base_url, selected_user_id)
                                    if sim_data and sim_data.get("similar_users"):
                                        sim_users = sim_data["similar_users"]
                                        df = pd.DataFrame(sim_users)
                                        st.dataframe(
                                            df[["name", "similarity", "shared_dishes"]],
                                            hide_index=True,
                                            use_container_width=True
                                        )
                    
                    # Show excluded dishes for users with dietary restrictions
                    excluded_data = api_get_excluded_dishes(api_base_url, selected_user_id)
                    if excluded_data and excluded_data.get("excluded_dishes"):
                        restrictions = excluded_data.get("restrictions", [])
                        excluded = excluded_data.get("excluded_dishes", [])
                        with st.expander(f"🚫 Dishes Excluded by Dietary Restrictions ({len(excluded)})"):
                            st.caption(f"Your restrictions: **{', '.join(restrictions)}**")
                            st.markdown("These dishes were filtered out because they contain unsuitable ingredients:")
                            for dish in excluded:
                                violations = dish.get("violations", [])
                                bad_ings = [v.get("ingredient", "unknown") for v in violations]
                                st.markdown(f"- **{dish['name']}** — contains: _{', '.join(bad_ings)}_")
    with tab2:
        st.header("📷 Search by Image")
        st.caption("Upload a food image to find similar dishes and their ingredients.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
            if st.button("Search Similar Dishes"):
                with st.spinner("Searching..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    data = api_search_by_image(api_base_url, uploaded_file)
                    
                if data and data.get("results"):
                    st.success(f"Found {data['count']} similar dishes!")
                    
                    for result in data["results"]:
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("Similarity", f"{result['score']:.2f}")
                                    
                            with col2:
                                st.subheader(result["name"])
                                if result.get("description"):
                                    st.write(result["description"])
                                
                                if result.get("ingredients"):
                                    st.markdown("**Ingredients:**")
                                    st.write(", ".join(result["ingredients"]))
                else:
                    st.info("No similar dishes found.")
