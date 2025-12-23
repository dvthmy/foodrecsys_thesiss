import numpy as np
import random
from src.services.recommendation_service import get_recommendation_service
from src.services.neo4j_service import Neo4jService
from src.visualization.similarity import (
    compute_jaccard_matrix,
    compute_image_embedding_matrix,
)

# Khởi tạo service
rec_service = get_recommendation_service()
neo4j = rec_service.neo4j


def calculate_metrics(recommended_ids, ground_truth_ids, k=10):
    """Tính Precision và Recall tại K."""
    if not ground_truth_ids or not recommended_ids:
        return 0.0, 0.0
    
    recommended_k = list(recommended_ids)[:k]
    hits = len(set(recommended_k) & set(ground_truth_ids))
    
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(ground_truth_ids)
    return precision, recall


def calculate_ndcg(recommended_ids, ground_truth_ids, k=10):
    """Calculate NDCG@K."""
    if not ground_truth_ids or not recommended_ids:
        return 0.0
    
    dcg = 0.0
    for i, dish_id in enumerate(list(recommended_ids)[:k]):
        if dish_id in ground_truth_ids:
            dcg += 1.0 / np.log2(i + 2)
    
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth_ids), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def get_similar_dishes_by_ingredients(user_liked_dish_ids, all_dishes_data, top_k=20):
    """Get similar dishes based on Jaccard ingredient similarity."""
    if not user_liked_dish_ids or not all_dishes_data:
        return []
    
    # Build ingredient sets
    dish_ingredients = {}
    for dish in all_dishes_data:
        dish_id = dish.get('dish_id')
        ingredients = set(dish.get('ingredients', []))
        if dish_id and ingredients:
            dish_ingredients[dish_id] = ingredients
    
    # Calculate similarity scores for each candidate dish
    scores = {}
    for candidate_id, candidate_ings in dish_ingredients.items():
        if candidate_id in user_liked_dish_ids:
            continue  # Skip dishes user already likes
        
        # Average Jaccard similarity to all liked dishes
        similarities = []
        for liked_id in user_liked_dish_ids:
            if liked_id in dish_ingredients:
                liked_ings = dish_ingredients[liked_id]
                intersection = len(candidate_ings & liked_ings)
                union = len(candidate_ings | liked_ings)
                if union > 0:
                    similarities.append(intersection / union)
        
        if similarities:
            scores[candidate_id] = np.mean(similarities)
    
    # Sort by score descending
    sorted_dishes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [dish_id for dish_id, score in sorted_dishes[:top_k]]


def get_similar_dishes_by_embedding(user_liked_dish_ids, all_dishes_data, top_k=20):
    """Get similar dishes based on image embedding similarity."""
    if not user_liked_dish_ids or not all_dishes_data:
        return []
    
    # Build embedding map
    dish_embeddings = {}
    for dish in all_dishes_data:
        dish_id = dish.get('dish_id')
        embedding = dish.get('image_embedding')
        if dish_id and embedding and len(embedding) > 0:
            dish_embeddings[dish_id] = np.array(embedding)
    
    # Calculate user profile (mean of liked dish embeddings)
    liked_embeddings = []
    for dish_id in user_liked_dish_ids:
        if dish_id in dish_embeddings:
            liked_embeddings.append(dish_embeddings[dish_id])
    
    if not liked_embeddings:
        return []
    
    user_profile = np.mean(liked_embeddings, axis=0)
    user_norm = np.linalg.norm(user_profile)
    
    if user_norm == 0:
        return []
    
    # Calculate similarity scores
    scores = {}
    for candidate_id, candidate_emb in dish_embeddings.items():
        if candidate_id in user_liked_dish_ids:
            continue
        
        candidate_norm = np.linalg.norm(candidate_emb)
        if candidate_norm > 0:
            cosine_sim = np.dot(user_profile, candidate_emb) / (user_norm * candidate_norm)
            scores[candidate_id] = cosine_sim
    
    sorted_dishes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [dish_id for dish_id, score in sorted_dishes[:top_k]]

# --- CÁC HÀM MỚI CHO HYBRID ---

def get_cf_scores(user_id, all_users, all_ratings, excluded_dish_ids):
    """
    Trả về Dict {dish_id: score} từ thuật toán CF.
    Logic giống hệt get_cf_recommendations nhưng trả về điểm số thô để tính toán tiếp.
    """
    user_ratings_map = {}
    for r in all_ratings:
        uid = r['user_id']
        did = r['dish_id']
        score = r['score']
        if uid not in user_ratings_map:
            user_ratings_map[uid] = {}
        user_ratings_map[uid][did] = score
    
    if user_id not in user_ratings_map:
        return {}
    
    target_ratings = user_ratings_map[user_id]
    target_dishes = set(target_ratings.keys())
    
    user_similarities = []
    for other_id, other_ratings in user_ratings_map.items():
        if other_id == user_id: continue
        
        common_dishes = target_dishes & set(other_ratings.keys())
        if len(common_dishes) < 1: continue
        
        target_vec = [target_ratings[d] for d in common_dishes]
        other_vec = [other_ratings[d] for d in common_dishes]
        
        if len(common_dishes) == 1:
            similarity = 1.0 if target_vec[0] == other_vec[0] else 0.5
        else:
            target_arr = np.array(target_vec) - np.mean(target_vec)
            other_arr = np.array(other_vec) - np.mean(other_vec)
            norm_t = np.linalg.norm(target_arr)
            norm_o = np.linalg.norm(other_arr)
            if norm_t > 0 and norm_o > 0:
                similarity = np.dot(target_arr, other_arr) / (norm_t * norm_o)
            else:
                similarity = 0.5
        
        if similarity > 0:
            user_similarities.append((other_id, similarity, other_ratings))
    
    if not user_similarities:
        return {}
    
    user_similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_users = user_similarities[:30]
    
    dish_scores = {}
    for other_id, sim, other_ratings in top_similar_users:
        for dish_id, rating in other_ratings.items():
            if dish_id in excluded_dish_ids: continue
            if rating >= 3:
                if dish_id not in dish_scores: dish_scores[dish_id] = []
                dish_scores[dish_id].append(sim * rating)
    
    # Trả về điểm trung bình
    return {did: np.mean(scores) for did, scores in dish_scores.items()}


def get_embedding_scores(user_liked_dish_ids, all_dishes_data):
    """
    Trả về Dict {dish_id: score} dùng Max-Pooling Strategy (Cách mới).
    """
    if not user_liked_dish_ids or not all_dishes_data:
        return {}
    
    dish_embeddings = {}
    for dish in all_dishes_data:
        dish_id = dish.get('dish_id')
        embedding = dish.get('image_embedding')
        if dish_id and embedding and len(embedding) > 0:
            vec = np.array(embedding)
            norm = np.linalg.norm(vec)
            if norm > 0:
                dish_embeddings[dish_id] = vec / norm 

    liked_embeddings = []
    for dish_id in user_liked_dish_ids:
        if dish_id in dish_embeddings:
            liked_embeddings.append(dish_embeddings[dish_id])
    
    if not liked_embeddings:
        return {}
        
    liked_matrix = np.array(liked_embeddings)
    scores = {}
    
    for candidate_id, candidate_emb in dish_embeddings.items():
        if candidate_id in user_liked_dish_ids: continue
        
        # Max-Pooling: So sánh với tất cả món đã like, lấy điểm cao nhất
        sim_scores = np.dot(liked_matrix, candidate_emb)
        max_score = np.max(sim_scores)
        scores[candidate_id] = max_score
        
    return scores


def get_hybrid_recommendations(user_id, all_users, all_ratings, all_dishes_data, train_dish_ids, top_k=20, alpha=0.7):
    """
    Kết hợp điểm số CF và Embedding theo công thức:
    Score = alpha * CF_Score + (1 - alpha) * Emb_Score
    """
    # 1. Lấy điểm số thô
    cf_scores = get_cf_scores(user_id, all_users, all_ratings, train_dish_ids)
    emb_scores = get_embedding_scores(train_dish_ids, all_dishes_data)
    
    all_candidates = set(cf_scores.keys()) | set(emb_scores.keys())
    if not all_candidates:
        return []

    # 2. Chuẩn hóa điểm số (Min-Max Normalization) về 0.0 - 1.0
    # Để tránh việc điểm CF (ví dụ 5.0) át hết điểm Cosine (ví dụ 0.8)
    def normalize_dict(score_dict):
        if not score_dict: return {}
        vals = score_dict.values()
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v: return {k: 1.0 for k in score_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}

    cf_norm = normalize_dict(cf_scores)
    emb_norm = normalize_dict(emb_scores)
    
    # 3. Cộng điểm
    final_scores = {}
    for did in all_candidates:
        s_cf = cf_norm.get(did, 0.0)
        s_emb = emb_norm.get(did, 0.0)
        
        # Công thức lai ghép
        final_scores[did] = (alpha * s_cf) + ((1 - alpha) * s_emb)
    
    # 4. Sắp xếp
    sorted_dishes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return [dish_id for dish_id, score in sorted_dishes[:top_k]]

def get_cf_recommendations(user_id, all_users, all_ratings, excluded_dish_ids, top_k=20):
    """Get CF recommendations based on similar users.
    
    Args:
        user_id: Target user ID
        all_users: List of all users
        all_ratings: List of all ratings
        excluded_dish_ids: Dishes to exclude (training set - already "seen")
        top_k: Number of recommendations
    """
    # Build user-dish rating matrix
    user_ratings_map = {}
    for r in all_ratings:
        uid = r['user_id']
        did = r['dish_id']
        score = r['score']
        if uid not in user_ratings_map:
            user_ratings_map[uid] = {}
        user_ratings_map[uid][did] = score
    
    if user_id not in user_ratings_map:
        return []
    
    target_ratings = user_ratings_map[user_id]
    target_dishes = set(target_ratings.keys())
    
    # Find similar users using Pearson correlation
    user_similarities = []
    for other_id, other_ratings in user_ratings_map.items():
        if other_id == user_id:
            continue
        
        # Find common dishes
        common_dishes = target_dishes & set(other_ratings.keys())
        if len(common_dishes) < 1:  # Relaxed from 2 to 1
            continue
        
        # Calculate cosine similarity on ratings
        target_vec = [target_ratings[d] for d in common_dishes]
        other_vec = [other_ratings[d] for d in common_dishes]
        
        if len(common_dishes) == 1:
            # For single common dish, use simple similarity
            similarity = 1.0 if target_vec[0] == other_vec[0] else 0.5
        else:
            target_arr = np.array(target_vec) - np.mean(target_vec)
            other_arr = np.array(other_vec) - np.mean(other_vec)
            
            norm_t = np.linalg.norm(target_arr)
            norm_o = np.linalg.norm(other_arr)
            
            if norm_t > 0 and norm_o > 0:
                similarity = np.dot(target_arr, other_arr) / (norm_t * norm_o)
            else:
                similarity = 0.5  # Default similarity if no variance
        
        if similarity > 0:
            user_similarities.append((other_id, similarity, other_ratings))
    
    if not user_similarities:
        return []
    
    # Sort by similarity
    user_similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar_users = user_similarities[:30]  # Increased from 20
    
    # Aggregate scores for dishes NOT in excluded set (can include test set)
    dish_scores = {}
    for other_id, sim, other_ratings in top_similar_users:
        for dish_id, rating in other_ratings.items():
            # Only exclude training dishes, NOT test dishes
            if dish_id in excluded_dish_ids:
                continue
            if rating >= 3:  # Lowered threshold from 4 to 3
                if dish_id not in dish_scores:
                    dish_scores[dish_id] = []
                dish_scores[dish_id].append(sim * rating)
    
    # Average weighted scores
    final_scores = {did: np.mean(scores) for did, scores in dish_scores.items()}
    sorted_dishes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [dish_id for dish_id, score in sorted_dishes[:top_k]]


def get_all_users_with_ratings(neo4j_service: Neo4jService):
    """
    Fetches all users who have rated at least one dish.
    """
    return neo4j_service.get_all_users_with_ratings()


def get_user_ratings(neo4j_service: Neo4jService, user_id: str):
    """
    Fetches all ratings for a specific user.
    """
    return neo4j_service.get_user_ratings(user_id)


def create_train_test_split(users, neo4j_service, num_users=100):
    """
    Creates a train/test split by hiding the highest-rated dish for a sample of users.
    """
    random_users = random.sample(users, min(num_users, len(users)))
    
    train_ratings = []
    test_ground_truth = {}

    all_ratings = neo4j_service.get_all_ratings()
    all_ratings_dict = {}
    for rating in all_ratings:
        uid = rating['user_id']
        if uid not in all_ratings_dict:
            all_ratings_dict[uid] = []
        all_ratings_dict[uid].append(rating)

    for user in random_users:
        user_id = user['user_id']
        user_ratings = all_ratings_dict.get(user_id, [])
        
        if not user_ratings:
            continue

        # Find the highest-rated dish
        highest_rated_dish = max(user_ratings, key=lambda x: x['rating'])
        
        # Store it as ground truth
        test_ground_truth[user_id] = [highest_rated_dish['dish_id']]
        
        # Create training data by excluding the hidden rating
        for rating in user_ratings:
            if rating['dish_id'] != highest_rated_dish['dish_id']:
                train_ratings.append(rating)

    # Add ratings from users not in the test set to the training set
    test_user_ids = set(test_ground_truth.keys())
    for user_id, ratings in all_ratings_dict.items():
        if user_id not in test_user_ids:
            train_ratings.extend(ratings)
            
    return train_ratings, test_ground_truth


def evaluate_cf_recommendations(rec_service, test_ground_truth, k=10):
    """
    Evaluates CF recommendations.
    """
    all_precisions = []
    all_recalls = []
    all_ndcgs = []

    for user_id, ground_truth_ids in test_ground_truth.items():
        # Get CF recommendations
        recommendations = rec_service.recommend_dishes(user_id, top_k=k)
        recommended_ids = [rec.dish_id for rec in recommendations]
        
        # Calculate metrics
        precision, recall = calculate_metrics(recommended_ids, ground_truth_ids, k)
        ndcg = calculate_ndcg(recommended_ids, ground_truth_ids, k)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_ndcgs.append(ndcg)

    # Calculate average metrics
    avg_precision = np.mean(all_precisions) if all_precisions else 0.0
    avg_recall = np.mean(all_recalls) if all_recalls else 0.0
    avg_ndcg = np.mean(all_ndcgs) if all_ndcgs else 0.0
    
    return avg_precision, avg_recall, avg_ndcg


def run_ablation_study():
    print("=" * 60)
    print("🧪 ABLATION STUDY: CF vs. CONTENT-BASED")
    print("=" * 60)

    # Load all data once
    print("Loading data...")
    all_dishes = neo4j.get_all_dishes_with_embeddings(limit=1000)
    all_dishes_ingredients = neo4j.get_all_dishes_ingredients()
    all_users = neo4j.get_all_users(limit=1000)
    all_ratings = neo4j.get_all_ratings()
    
    # Merge dish data
    dish_data_map = {}
    for dish in all_dishes_ingredients:
        dish_data_map[dish['dish_id']] = dish
    for dish in all_dishes:
        if dish['dish_id'] in dish_data_map:
            dish_data_map[dish['dish_id']]['image_embedding'] = dish.get('image_embedding')
        else:
            dish_data_map[dish['dish_id']] = dish
    
    all_dishes_data = list(dish_data_map.values())
    all_dish_ids = set(dish_data_map.keys())
    
    print(f"Total dishes: {len(all_dish_ids)}")
    print(f"Dishes with embeddings: {sum(1 for d in all_dishes_data if d.get('image_embedding'))}")
    print(f"Total users: {len(all_users)}")
    print(f"Total ratings: {len(all_ratings)}")

    # Build user ratings map
    user_ratings_map = {}
    for r in all_ratings:
        uid = r['user_id']
        if uid not in user_ratings_map:
            user_ratings_map[uid] = []
        user_ratings_map[uid].append(r)

    # Filter users with enough ratings
    test_users = [u for u in all_users if len(user_ratings_map.get(u['user_id'], [])) >= 5]
    print(f"Testing on {len(test_users)} users with >= 5 ratings...")

    # Store results
    
        # ... code cũ ...
    metrics = {
        "Random": {"p10": [], "r10": [], "ndcg10": [], "hits": 0},
        "CF": {"p10": [], "r10": [], "ndcg10": [], "hits": 0},
        "CBF_Jaccard": {"p10": [], "r10": [], "ndcg10": [], "hits": 0},
        "CBF_Embedding": {"p10": [], "r10": [], "ndcg10": [], "hits": 0},
        "Hybrid": {"p10": [], "r10": [], "ndcg10": [], "hits": 0}  # <--- THÊM DÒNG NÀY
    }
    # ...
    
    
    evaluated_users = 0
    debug_count = 0
    total_ground_truth_size = 0

    for idx, user in enumerate(test_users):
        user_id = user['user_id']
        user_ratings = user_ratings_map.get(user_id, [])
        
        # Split: liked dishes (rating >= 4)
        liked_dishes = [r for r in user_ratings if r['score'] >= 4]
        
        if len(liked_dishes) < 2:
            continue
        
        # Hold-out: 30% as test set
        n_test = max(1, len(liked_dishes) // 3)
        random.shuffle(liked_dishes)
        
        test_set = liked_dishes[:n_test]
        train_set = liked_dishes[n_test:]
        
        ground_truth = {d['dish_id'] for d in test_set}
        train_dish_ids = {d['dish_id'] for d in train_set}
        
        if not train_dish_ids or not ground_truth:
            continue
        
        evaluated_users += 1
        total_ground_truth_size += len(ground_truth)
        
        # Get candidate dishes (all dishes except training set)
        candidate_dishes = list(all_dish_ids - train_dish_ids)
        
        # Debug first few users
        if debug_count < 3:
            print(f"\n  [DEBUG] User {user_id}: train={len(train_dish_ids)}, test={len(ground_truth)}")
        
        # --- TEST 0: RANDOM BASELINE ---
        random_recs = random.sample(candidate_dishes, min(20, len(candidate_dishes)))
        random_hits = set(random_recs) & ground_truth
        if random_hits:
            metrics["Random"]["hits"] += 1
        
        p, r = calculate_metrics(random_recs, ground_truth, k=10)
        ndcg = calculate_ndcg(random_recs, ground_truth, k=10)
        metrics["Random"]["p10"].append(p)
        metrics["Random"]["r10"].append(r)
        metrics["Random"]["ndcg10"].append(ndcg)
        
        # --- TEST 1: COLLABORATIVE FILTERING ---
        # Pass training dish IDs as excluded (so CF can recommend test dishes)
        cf_recs = get_cf_recommendations(user_id, all_users, all_ratings, 
                                         excluded_dish_ids=train_dish_ids, top_k=20)
        # Check if any ground truth in recommendations
        cf_hits = set(cf_recs) & ground_truth
        if cf_hits:
            metrics["CF"]["hits"] += 1
        
        p, r = calculate_metrics(cf_recs, ground_truth, k=10)
        ndcg = calculate_ndcg(cf_recs, ground_truth, k=10)
        metrics["CF"]["p10"].append(p)
        metrics["CF"]["r10"].append(r)
        metrics["CF"]["ndcg10"].append(ndcg)
        
        if debug_count < 3:
            print(f"    CF: {len(cf_recs)} recs, hits={len(cf_hits)}, P@10={p:.3f}")

        # --- TEST 2: CONTENT-BASED (JACCARD) ---
        cbf_j_recs = get_similar_dishes_by_ingredients(train_dish_ids, all_dishes_data, top_k=20)
        cbf_j_hits = set(cbf_j_recs) & ground_truth
        if cbf_j_hits:
            metrics["CBF_Jaccard"]["hits"] += 1
        
        p, r = calculate_metrics(cbf_j_recs, ground_truth, k=10)
        ndcg = calculate_ndcg(cbf_j_recs, ground_truth, k=10)
        metrics["CBF_Jaccard"]["p10"].append(p)
        metrics["CBF_Jaccard"]["r10"].append(r)
        metrics["CBF_Jaccard"]["ndcg10"].append(ndcg)
        
        if debug_count < 3:
            print(f"    Jaccard: {len(cbf_j_recs)} recs, hits={len(cbf_j_hits)}, P@10={p:.3f}")

        # --- TEST 3: CONTENT-BASED (EMBEDDING) ---
        cbf_e_recs = get_similar_dishes_by_embedding(train_dish_ids, all_dishes_data, top_k=20)
        cbf_e_hits = set(cbf_e_recs) & ground_truth
        if cbf_e_hits:
            metrics["CBF_Embedding"]["hits"] += 1
        
        p, r = calculate_metrics(cbf_e_recs, ground_truth, k=10)
        ndcg = calculate_ndcg(cbf_e_recs, ground_truth, k=10)
        metrics["CBF_Embedding"]["p10"].append(p)
        metrics["CBF_Embedding"]["r10"].append(r)
        metrics["CBF_Embedding"]["ndcg10"].append(ndcg)
        
        if debug_count < 3:
            print(f"    Embedding: {len(cbf_e_recs)} recs, hits={len(cbf_e_hits)}, P@10={p:.3f}")
            
    

        # --- TEST 4: HYBRID (CF + EMBEDDING) ---
        # Alpha = 0.7 nghĩa là tin tưởng CF 70%, Embedding 30%
        hybrid_recs = get_hybrid_recommendations(
            user_id, all_users, all_ratings, all_dishes_data, 
            train_dish_ids=train_dish_ids, 
            top_k=20, 
            alpha=0.7 
        )
        hybrid_hits = set(hybrid_recs) & ground_truth
        if hybrid_hits:
            metrics["Hybrid"]["hits"] += 1
        
        p, r = calculate_metrics(hybrid_recs, ground_truth, k=10)
        ndcg = calculate_ndcg(hybrid_recs, ground_truth, k=10)
        metrics["Hybrid"]["p10"].append(p)
        metrics["Hybrid"]["r10"].append(r)
        metrics["Hybrid"]["ndcg10"].append(ndcg)
        
        if debug_count < 3:
            print(f"    Hybrid: {len(hybrid_recs)} recs, hits={len(hybrid_hits)}, P@10={p:.3f}")
        
        debug_count += 1
        
        # Progress indicator
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(test_users)} users...")
            


    print(f"\n✅ Successfully evaluated {evaluated_users} users")
    print(f"📈 Average ground truth size: {total_ground_truth_size / evaluated_users:.2f} items")
    
    # Calculate theoretical random baseline
    avg_gt = total_ground_truth_size / evaluated_users
    n_dishes = len(all_dish_ids)
    theoretical_random_precision = avg_gt / n_dishes
    print(f"📊 Theoretical random P@10: {theoretical_random_precision:.4f}")
    
    # Display results
    print("\n" + "=" * 75)
    print("📊 KẾT QUẢ TRUNG BÌNH (Mean Metrics):")
    print("=" * 75)
    print(f"{'Method':<20} | {'Precision@10':<12} | {'Recall@10':<12} | {'NDCG@10':<12} | {'Hit Users':<10}")
    print("-" * 75)
    
    for method, scores in metrics.items():
        if scores["p10"]:
            avg_p = np.mean(scores["p10"])
            avg_r = np.mean(scores["r10"])
            avg_ndcg = np.mean(scores["ndcg10"])
            hits = scores["hits"]
            print(f"{method:<20} | {avg_p:<12.4f} | {avg_r:<12.4f} | {avg_ndcg:<12.4f} | {hits:<10}")
        else:
            print(f"{method:<20} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<10}")
    
    print("-" * 75)
    
    print("\n📝 INTERPRETATION:")
    print("  • Precision@10 = Fraction of top-10 recommendations that are relevant")
    print("  • Recall@10 = Fraction of relevant items found in top-10")
    print("  • NDCG@10 = Ranking quality (relevant items ranked higher is better)")
    print("  • Hit Users = Number of users where at least 1 ground truth item was recommended")


def main():
    """Evaluate CF model using train/test split on 100 random users."""
    print("\n" + "=" * 60)
    print("COLLABORATIVE FILTERING EVALUATION (Train/Test Split)")
    print("=" * 60)
    
    # 1. Get all users and ratings
    users = get_all_users_with_ratings(neo4j)
    if not users:
        print("No users with ratings found.")
        return

    all_ratings = neo4j.get_all_ratings()
    if not all_ratings:
        print("No ratings found.")
        return

    # Build ratings map per user
    user_ratings_map = {}
    for r in all_ratings:
        uid = r['user_id']
        if uid not in user_ratings_map:
            user_ratings_map[uid] = []
        user_ratings_map[uid].append(r)

    # 2. Filter users who have at least 2 ratings (need 1 for test, 1 for train)
    eligible_users = [u for u in users if len(user_ratings_map.get(u['user_id'], [])) >= 2]
    
    if len(eligible_users) < 10:
        print(f"Not enough eligible users (found {len(eligible_users)}, need at least 10)")
        return

    # 3. Sample 100 random users
    num_test_users = min(100, len(eligible_users))
    test_users = random.sample(eligible_users, num_test_users)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total users with ratings: {len(users)}")
    print(f"  Eligible users (≥2 ratings): {len(eligible_users)}")
    print(f"  Test users sampled: {num_test_users}")
    print(f"  Total ratings: {len(all_ratings)}")

    # 4. Evaluate each user
    all_precisions = []
    all_recalls = []
    all_ndcgs = []
    hit_users = 0
    k = 10

    for user in test_users:
        user_id = user['user_id']
        user_ratings = user_ratings_map.get(user_id, [])
        
        if len(user_ratings) < 2:
            continue

        # Find the highest-rated dish as ground truth
        highest_rated = max(user_ratings, key=lambda x: x['rating'])
        ground_truth_ids = [highest_rated['dish_id']]
        
        # Training set excludes the hidden dish
        train_dish_ids = {r['dish_id'] for r in user_ratings if r['dish_id'] != highest_rated['dish_id']}
        
        # Get CF recommendations (excludes training dishes)
        recommended_ids = get_cf_recommendations(
            user_id=user_id,
            all_users=users,
            all_ratings=all_ratings,
            excluded_dish_ids=train_dish_ids,
            top_k=k
        )
        
        # Calculate metrics
        precision, recall = calculate_metrics(recommended_ids, ground_truth_ids, k)
        ndcg = calculate_ndcg(recommended_ids, ground_truth_ids, k)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_ndcgs.append(ndcg)
        
        if set(recommended_ids[:k]) & set(ground_truth_ids):
            hit_users += 1

    # 5. Calculate averages
    evaluated_users = len(all_precisions)
    avg_precision = np.mean(all_precisions) if all_precisions else 0.0
    avg_recall = np.mean(all_recalls) if all_recalls else 0.0
    avg_ndcg = np.mean(all_ndcgs) if all_ndcgs else 0.0
    hit_rate = hit_users / evaluated_users if evaluated_users > 0 else 0.0

    print(f"\n📈 RESULTS (Collaborative Filtering):")
    print(f"  Users evaluated: {evaluated_users}")
    print(f"  Average Precision@{k}: {avg_precision:.4f}")
    print(f"  Average Recall@{k}:    {avg_recall:.4f}")
    print(f"  Average NDCG@{k}:      {avg_ndcg:.4f}")
    print(f"  Hit Rate:              {hit_rate:.4f} ({hit_users}/{evaluated_users} users)")
    
    print("\n📝 INTERPRETATION:")
    print("  • Hit Rate = Fraction of users where the hidden dish was recommended")
    print("  • Higher values indicate better recommendation quality")


if __name__ == "__main__":
    random.seed(42)
    run_ablation_study()
    main()