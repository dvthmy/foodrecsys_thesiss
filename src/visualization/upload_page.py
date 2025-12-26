"""Streamlit page for uploading and processing dishes via API.

Provides UI for:
- Uploading dish images (single or batch)
- Adding dish names and descriptions
- Monitoring processing job status
- Viewing processing results
"""

import os
import time
from typing import Any

import requests
import streamlit as st
from PIL import Image
import io


def api_upload_and_process(
    api_base_url: str,
    images: list[tuple[str, bytes, str]],
    names: list[str],
    descriptions: list[str],
) -> dict:
    """Upload images and start processing via API.
    
    Args:
        api_base_url: Base URL for the FastAPI server
        images: List of (filename, content, content_type) tuples
        names: List of dish names
        descriptions: List of dish descriptions
        
    Returns:
        API response dict
    """
    url = f"{api_base_url.rstrip('/')}/dishes/upload-and-process"
    
    files = [
        ("images", (filename, content, content_type))
        for filename, content, content_type in images
    ]
    
    data = {}
    if names:
        data["names"] = names
    if descriptions:
        data["descriptions"] = descriptions
    
    response = requests.post(url, files=files, data=data, timeout=60)
    response.raise_for_status()
    return response.json()


def api_get_job_status(api_base_url: str, job_id: str) -> dict:
    """Get job status from API.
    
    Args:
        api_base_url: Base URL for the FastAPI server
        job_id: The job ID to check
        
    Returns:
        Job status dict
    """
    url = f"{api_base_url.rstrip('/')}/jobs/{job_id}/status"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def render_upload_form() -> tuple[list, list[str], list[str]]:
    """Render the upload form and return collected data.
    
    Returns:
        Tuple of (uploaded_files, names, descriptions)
    """
    st.subheader("📤 Upload Dish Images")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose dish images",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        accept_multiple_files=True,
        help="Upload one or more dish images. Supported formats: JPG, PNG, GIF, WebP",
    )
    
    names: list[str] = []
    descriptions: list[str] = []
    
    if uploaded_files:
        st.markdown("---")
        st.subheader("📝 Dish Details")
        st.markdown(
            "Provide names and descriptions for each dish. "
            "**Descriptions are important for ingredient extraction.**"
        )
        
        # Create input fields for each uploaded file
        for i, file in enumerate(uploaded_files):
            with st.expander(f"📷 {file.name}", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Show image preview
                    try:
                        image = Image.open(file)
                        st.image(image, width=200, caption=file.name)
                        # Reset file pointer for later reading
                        file.seek(0)
                    except Exception as e:
                        st.error(f"Cannot preview image: {e}")
                
                with col2:
                    # Name input
                    name = st.text_input(
                        "Dish Name",
                        key=f"name_{i}",
                        placeholder="e.g., Pad Thai, Pho, Spaghetti Carbonara",
                        help="Enter the name of the dish",
                    )
                    names.append(name)
                    
                    # Description input
                    description = st.text_area(
                        "Description (for ingredient extraction)",
                        key=f"desc_{i}",
                        placeholder="Describe the dish, including its main ingredients...\n\nExample: A classic Vietnamese noodle soup with rice noodles, beef broth, thinly sliced beef, bean sprouts, fresh herbs like basil and cilantro, and lime wedges.",
                        help="Provide a detailed description including ingredients. This is used by Gemma 3 for ingredient extraction.",
                        height=120,
                    )
                    descriptions.append(description)
    
    return uploaded_files, names, descriptions


def render_job_status(api_base_url: str, job_id: str) -> dict | None:
    """Render job status with auto-refresh.
    
    Args:
        api_base_url: Base URL for the FastAPI server
        job_id: The job ID to monitor
        
    Returns:
        Final job status dict or None if still processing
    """
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    while True:
        try:
            job_status = api_get_job_status(api_base_url, job_id)
        except requests.exceptions.ReadTimeout:
            st.warning(
                "⏱️ API timeout when getting job status. "
                "The server may be restarting or busy. Please try again in a few seconds."
            )
            return None
        except Exception as e:
            st.error(f"Failed to get job status: {e}")
            return None
        
        status = job_status.get("status", "unknown")
        progress = job_status.get("progress", 0)
        total = job_status.get("total_items", 0)
        completed = job_status.get("completed", 0)
        failed = job_status.get("failed", 0)
        
        # Update progress bar
        progress_bar.progress(min(progress, 1.0))
        
        # Status display
        with status_container.container():
            if status == "processing":
                st.info(f"⏳ Processing... {completed}/{total} items ({progress*100:.0f}%)")
            elif status == "completed":
                st.success(f"✅ Completed! {completed}/{total} items processed successfully.")
                if failed > 0:
                    st.warning(f"⚠️ {failed} items failed.")
            elif status == "failed":
                st.error(f"❌ Job failed. {failed}/{total} items failed.")
            else:
                st.info(f"Status: {status}")
        
        # If job is done, break the loop
        if status in ["completed", "failed"]:
            return job_status
        
        # Wait before next poll (reduced call frequency to ease API load)
        time.sleep(2)


def render_job_results(job_status: dict) -> None:
    """Render the results of a completed job.
    
    Args:
        job_status: The final job status dict
    """
    st.markdown("---")
    st.subheader("📊 Processing Results")
    
    results = job_status.get("results", {})
    
    # Handle both dict and list formats
    if isinstance(results, dict):
        results_list = list(results.values())
    else:
        results_list = results
    
    if not results_list:
        st.info("No results available.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    success_count = sum(1 for r in results_list if isinstance(r, dict) and r.get("success", False))
    failed_count = len(results_list) - success_count
    
    with col1:
        st.metric("Total Items", len(results_list))
    with col2:
        st.metric("Successful", success_count, delta=None)
    with col3:
        st.metric("Failed", failed_count, delta=None if failed_count == 0 else f"-{failed_count}")
    
    # Detailed results
    for i, result in enumerate(results_list):
        if not isinstance(result, dict):
            continue
            
        success = result.get("success", False)
        dish_name = result.get("dish_name") or result.get("name") or result.get("original_name", f"Item {i+1}")
        
        icon = "✅" if success else "❌"
        
        with st.expander(f"{icon} {dish_name}", expanded=not success):
            if success:
                # Show extracted ingredients
                ingredients = result.get("ingredients", [])
                if ingredients:
                    st.markdown("**Extracted Ingredients:**")
                    cols = st.columns(3)
                    for j, ing in enumerate(ingredients):
                        cols[j % 3].markdown(f"• {ing}")
                else:
                    st.info("No ingredients extracted.")
                
                # Show dish ID if available
                dish_id = result.get("dish_id")
                if dish_id:
                    st.caption(f"Dish ID: `{dish_id}`")
                    
                # Show embedding status
                has_embedding = result.get("has_embedding", False)
                if has_embedding:
                    st.caption("✅ Image embedding generated")
            else:
                # Show error
                error = result.get("error", "Unknown error")
                st.error(f"Error: {error}")


def render_upload_page(api_base_url: str) -> None:
    """Render the main upload page.
    
    Args:
        api_base_url: Base URL for the FastAPI server
    """
    st.title("🍜 Upload & Process Dishes")
    st.markdown(
        "Upload dish images with descriptions to extract ingredients and store in the database. "
        "The system uses **Gemma 3** for ingredient extraction from text and **CLIP** for image embeddings."
    )
    
    # Check if there's an ongoing job in session state
    if "current_job_id" in st.session_state and st.session_state.current_job_id:
        st.info(f"📋 Monitoring job: `{st.session_state.current_job_id}`")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        with col2:
            if st.button("🗑️ Clear & Upload New"):
                st.session_state.current_job_id = None
                st.rerun()
        
        # Show job status
        job_status = render_job_status(api_base_url, st.session_state.current_job_id)
        
        if job_status:
            render_job_results(job_status)
            
            # Option to start new upload
            if st.button("📤 Upload More Dishes"):
                st.session_state.current_job_id = None
                st.rerun()
        return
    
    # Render upload form
    uploaded_files, names, descriptions = render_upload_form()
    
    if not uploaded_files:
        st.info("👆 Upload one or more dish images to get started.")
        
        # Show example/instructions
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            ### Processing Pipeline
            
            1. **Upload**: Select one or more dish images
            2. **Describe**: Add names and descriptions for each dish
            3. **Process**: The system will:
               - Extract ingredients from descriptions using **Gemma 3**
               - Generate image embeddings using **CLIP**
               - Store everything in **Neo4j** database
            4. **Review**: Check extracted ingredients and processing results
            
            """)
        return
    
    # Validation
    missing_descriptions = [
        names[i] or uploaded_files[i].name
        for i in range(len(uploaded_files))
        if not descriptions[i].strip()
    ]
    
    if missing_descriptions:
        st.warning(
            f"⚠️ Missing descriptions for: {', '.join(missing_descriptions[:3])}"
            f"{'...' if len(missing_descriptions) > 3 else ''}. "
            "Descriptions are required for ingredient extraction."
        )
    
    # Submit button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        submit_disabled = not all(d.strip() for d in descriptions)
        
        if st.button(
            "🚀 Upload & Process",
            type="primary",
            use_container_width=True,
            disabled=submit_disabled,
        ):
            if not api_base_url:
                st.error("Please configure the API base URL in the sidebar.")
                return
            
            # Prepare files for upload
            images_data: list[tuple[str, bytes, str]] = []
            names_to_send: list[str] = []
            for file, name in zip(uploaded_files, names):
                content = file.read()
                content_type = file.type or "image/jpeg"

                # Preserve original filename with extension so the API passes allowed_file checks
                filename = file.name

                # Fallback to filename stem if the user left the name blank
                clean_name = name.strip() or os.path.splitext(file.name)[0]

                images_data.append((filename, content, content_type))
                names_to_send.append(clean_name)
            
            try:
                with st.spinner("Uploading and starting processing..."):
                    response = api_upload_and_process(
                        api_base_url,
                        images_data,
                        names_to_send,
                        descriptions,
                    )
                
                job_id = response.get("job_id")
                if job_id:
                    st.session_state.current_job_id = job_id
                    st.success(f"✅ Upload successful! Job ID: `{job_id}`")
                    
                    # Show upload errors if any
                    upload_errors = response.get("upload_errors", [])
                    if upload_errors:
                        st.warning(f"⚠️ {len(upload_errors)} file(s) had upload errors:")
                        for err in upload_errors:
                            st.error(f"  • {err.get('filename')}: {err.get('error')}")
                    
                    st.rerun()
                else:
                    st.error("No job ID returned from API.")
                    
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Cannot connect to API at `{api_base_url}`. "
                    "Please check if the server is running."
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API error: {e.response.text if e.response else str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
        
        if submit_disabled:
            st.caption("Please fill in all descriptions to enable processing.")


def render_batch_upload_page(api_base_url: str) -> None:
    """Render batch upload page with CSV/JSON support.
    
    Args:
        api_base_url: Base URL for the FastAPI server
    """
    st.title("📦 Batch Upload")
    st.markdown("Upload multiple dishes at once using a metadata file.")
    
    st.info("🚧 Batch upload with metadata file coming soon!")
    
    # Placeholder for future implementation
    with st.expander("📋 Expected Format"):
        st.markdown("""
        ### JSON Format
        ```json
        {
            "dishes": [
                {
                    "name": "Pad Thai",
                    "description": "Thai stir-fried noodles with...",
                    "image_path": "images/pad_thai.jpg"
                }
            ]
        }
        ```
        
        ### CSV Format
        ```csv
        name,description,image_path
        Pad Thai,"Thai stir-fried noodles with...",images/pad_thai.jpg
        ```
        """)
