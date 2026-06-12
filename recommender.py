import joblib
import numpy as np

# Load model dan metadata dari file .pkl saat backend di-start
# File global_bite_model.pkl harus diletakkan satu folder dengan file ini
try:
    model = joblib.load('global_bite_model.pkl')
    df_unique_menu = model['df_unique_menu']
    df_final = model['df_final']
    cosine_sim_matrix = model['cosine_sim_matrix']
    tfidf = model['tfidf_vectorizer']
    tfidf_matrix = model['tfidf_matrix']
    TAG_KEYWORDS = model['TAG_KEYWORDS']
    VALID_TAGS = model['VALID_TAGS']
    print("✅ Model global_bite_model.pkl loaded successfully!")
except FileNotFoundError:
    print("⚠️ WARNING: global_bite_model.pkl tidak ditemukan! Silakan salin file .pkl ke folder ini.")
    df_unique_menu = None
    df_final = None 
    cosine_sim_matrix = None
    tfidf = None
    tfidf_matrix = None
    TAG_KEYWORDS = {}
    VALID_TAGS = []

def compute_tag_overlap(tags_a, tags_b):
    """Menghitung Jaccard Similarity antara dua set tags rasa."""
    set_a, set_b = set(tags_a), set(tags_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

def recommend_cross_city(liked_menus, target_city, top_n=5):
    """
    Rekomendasi cross-city berdasarkan menu yang pernah disukai user.
    Misal: user suka Bakso di Jakarta -> rekomendasikan menu serupa di Tokyo.
    """
    if df_unique_menu is None:
        return {'error': 'Model belum ter-load sempurna. global_bite_model.pkl tidak ditemukan.'}
        
    liked_indices = []
    for menu in liked_menus:
        m = df_unique_menu[df_unique_menu['menu_name'].str.lower() == menu.lower()]
        if not m.empty:
            liked_indices.append(m.index[0])

    if not liked_indices:
        return {'error': 'Tidak ada menu yang ditemukan dari daftar liked_menus.'}

    # User taste profile: rata-rata TF-IDF vector (dibungkus asarray agar kompatibel dengan sklearn/numpy versi baru)
    user_vector = np.asarray(tfidf_matrix[liked_indices].mean(axis=0))

    # User tag profile: union semua tags rasa dari menu yang disukai
    user_tags = set()
    for idx in liked_indices:
        user_tags.update(df_unique_menu.loc[idx, 'flavor_tags'])

    # Filter menu yang ada di target city
    city_mask = df_unique_menu['city_name'].str.lower() == target_city.lower()
    city_menus = df_unique_menu[city_mask].copy()

    if city_menus.empty:
        return {'error': f"Kota '{target_city}' tidak ditemukan dalam dataset."}

    city_indices = city_menus.index.tolist()

    # Hitung TF-IDF similarity (kemiripan deskripsi & kategori)
    city_tfidf = tfidf_matrix[city_indices]
    from sklearn.metrics.pairwise import cosine_similarity
    tfidf_sims = cosine_similarity(user_vector, city_tfidf).flatten()

    # Hitung tag overlap (kemiripan profil rasa)
    tag_overlaps = []
    for idx in city_indices:
        menu_tags = set(df_unique_menu.loc[idx, 'flavor_tags'])
        overlap = compute_tag_overlap(user_tags, menu_tags)
        tag_overlaps.append(overlap)
    tag_overlaps = np.array(tag_overlaps)

    # Gabungkan sinyal kemiripan (bobot: 70% TF-IDF, 30% Jaccard Overlap)
    combined_sim = (tfidf_sims * 0.7) + (tag_overlaps * 0.3)

    # Normalisasi Popularity Score lokal
    pop_scores = city_menus['popularity_score'].values
    pop_min, pop_max = pop_scores.min(), pop_scores.max()
    norm_pop = (pop_scores - pop_min) / (pop_max - pop_min) if pop_max > pop_min else np.full_like(pop_scores, 0.5)

    # Hitung skor akhir (bobot: 60% kemiripan rasa, 40% popularitas restoran)
    final_scores = (combined_sim * 0.6) + (norm_pop * 0.4)

    # Susun output rekomendasi (kecualikan menu yang sudah di-like)
    liked_lower = {m.lower() for m in liked_menus}
    scored = list(zip(range(len(city_menus)), final_scores, combined_sim, tag_overlaps))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, fscore, csim, toverlap in scored:
        row = city_menus.iloc[i]
        if row['menu_name'].lower() in liked_lower:
            continue
            
        results.append({
            'menu_name': row['menu_name'],
            'menu_category': row['menu_category'],
            'menu_description': row['menu_description'],
            'menu_image_url': row.get('menu_image_url', ''),
            'menu_avg_rating': float(row['menu_avg_rating']),
            'menu_review_count': int(row['menu_review_count']),
            'flavor_tags': row['flavor_tags'],
            'city_name': row['city_name'],
            'country_name': row.get('country_name', ''),
            'currency_code': row.get('currency_code', ''),
            'currency_symbol': row.get('currency_symbol', ''),
            'similarity_score': round(float(csim), 4),
            'popularity_score': round(float(row['popularity_score']), 4),
            'final_score': round(float(fscore), 4),
        })
        if len(results) >= top_n:
            break

    return results

def recommend_by_tags(user_tags, target_city, top_n=5):
    """Rekomendasi berdasarkan preference tags pilihan user (untuk user baru)."""
    if df_unique_menu is None:
        return {'error': 'Model belum ter-load sempurna. global_bite_model.pkl tidak ditemukan.'}
        
    valid = [t for t in user_tags if t in VALID_TAGS]
    if not valid:
        return {'error': f'Tidak ada tag valid. Tags yang tersedia: {VALID_TAGS}'}

    city_mask = df_unique_menu['city_name'].str.lower() == target_city.lower()
    city_menus = df_unique_menu[city_mask].copy()

    if city_menus.empty:
        return {'error': f"Kota '{target_city}' tidak ditemukan dalam dataset."}

    # Hitung kecocokan rasa pilihan user dengan rasa makanan di kota tujuan
    user_tag_set = set(valid)
    tag_scores = []
    for _, row in city_menus.iterrows():
        menu_tags = set(row['flavor_tags'])
        matched = len(user_tag_set & menu_tags)
        score = matched / len(user_tag_set) if user_tag_set else 0
        tag_scores.append(score)

    city_menus['tag_match_score'] = tag_scores

    # Normalisasi Popularity Score
    pop_scores = city_menus['popularity_score'].values
    pop_min, pop_max = pop_scores.min(), pop_scores.max()
    norm_pop = (pop_scores - pop_min) / (pop_max - pop_min) if pop_max > pop_min else np.full_like(pop_scores, 0.5)
    city_menus['norm_popularity'] = norm_pop

    # Hitung skor akhir
    city_menus['final_score'] = (city_menus['tag_match_score'] * 0.6) + (city_menus['norm_popularity'] * 0.4)

    top = city_menus.nlargest(top_n, 'final_score')

    results = []
    for _, row in top.iterrows():
        results.append({
            'menu_name': row['menu_name'],
            'menu_category': row['menu_category'],
            'menu_description': row['menu_description'],
            'menu_image_url': row.get('menu_image_url', ''),
            'menu_avg_rating': float(row['menu_avg_rating']),
            'menu_review_count': int(row['menu_review_count']),
            'flavor_tags': row['flavor_tags'],
            'city_name': row['city_name'],
            'country_name': row.get('country_name', ''),
            'currency_code': row.get('currency_code', ''),
            'currency_symbol': row.get('currency_symbol', ''),
            'tag_match_score': round(float(row['tag_match_score']), 4),
            'popularity_score': round(float(row['popularity_score']), 4),
            'final_score': round(float(row['final_score']), 4),
        })

    return results

def get_top_by_city(city_name, top_n=5):
    """Menu terpopuler di kota tertentu (fallback jika tidak ada preferensi)."""
    if df_final is None:
        return {'error': 'Model belum ter-load sempurna. global_bite_model.pkl tidak ditemukan.'}
        
    city_mask = df_final['city_name'].str.lower() == city_name.lower()
    city_data = df_final[city_mask]

    if city_data.empty:
        return {'error': f"Kota '{city_name}' tidak ditemukan dalam dataset."}

    top = city_data.sort_values('popularity_score', ascending=False).drop_duplicates('menu_name').head(top_n)

    results = []
    for _, row in top.iterrows():
        results.append({
            'menu_name': row['menu_name'],
            'menu_category': row['menu_category'],
            'menu_description': row['menu_description'],
            'menu_image_url': row.get('menu_image_url', ''),
            'menu_avg_rating': float(row['menu_avg_rating']),
            'menu_review_count': int(row['menu_review_count']),
            'flavor_tags': row['flavor_tags'],
            'merchant_name': row['merchant_name'],
            'merchant_address': row.get('merchant_address', ''),
            'price_min': float(row.get('price_min', 0)),
            'price_max': float(row.get('price_max', 0)),
            'opening_time': str(row.get('opening_time', '')),
            'closing_time': str(row.get('closing_time', '')),
            'city_name': row['city_name'],
            'country_name': row.get('country_name', ''),
            'currency_code': row.get('currency_code', ''),
            'currency_symbol': row.get('currency_symbol', ''),
            'popularity_score': round(float(row['popularity_score']), 4),
        })

    return results

def get_similar_menus(menu_name, top_n=5):
    """Rekomendasi menu serupa secara global (Content-Based murni)."""
    if df_unique_menu is None:
        return {'error': 'Model belum ter-load sempurna. global_bite_model.pkl tidak ditemukan.'}
        
    matches = df_unique_menu[df_unique_menu['menu_name'].str.lower() == menu_name.lower()]
    if matches.empty:
        return {'error': f"Menu '{menu_name}' tidak ditemukan."}

    idx = matches.index[0]
    input_tags = df_unique_menu.loc[idx, 'flavor_tags']
    tfidf_scores = cosine_sim_matrix[idx]

    results = []
    for i in range(len(df_unique_menu)):
        if i == idx:
            continue
        tag_sim = compute_tag_overlap(input_tags, df_unique_menu.loc[i, 'flavor_tags'])
        combined = (tfidf_scores[i] * 0.7) + (tag_sim * 0.3)
        results.append((i, combined, tfidf_scores[i], tag_sim))

    results.sort(key=lambda x: x[1], reverse=True)

    output = []
    for i, combined, tfidf_s, tag_s in results[:top_n]:
        row = df_unique_menu.iloc[i]
        output.append({
            'menu_name': row['menu_name'],
            'menu_category': row['menu_category'],
            'city_name': row['city_name'],
            'country_name': row.get('country_name', ''),
            'currency_code': row.get('currency_code', ''),
            'currency_symbol': row.get('currency_symbol', ''),
            'flavor_tags': row['flavor_tags'],
            'combined_score': round(combined, 4),
            'tfidf_score': round(tfidf_s, 4),
            'tag_overlap': round(tag_s, 4),
        })

    return output

def get_available_cities():
    """Mengembalikan daftar kota dan negara yang tersedia di dataset."""
    if df_unique_menu is None:
        return []
     # Ambil kombinasi unik kota, negara, dan mata uang
    cols = ['city_name', 'country_name', 'currency_code', 'currency_symbol']
    available_cols = [c for c in cols if c in df_unique_menu.columns]
    cities_df = df_unique_menu[available_cols].drop_duplicates()
    return cities_df.to_dict(orient='records')

def get_menu_detail(menu_name, city_name):
    """Mendapatkan detail menu beserta daftar semua merchant/restoran yang menjualnya di kota tersebut."""
    if df_final is None:
        return {'error': 'Model belum ter-load sempurna. global_bite_model.pkl tidak ditemukan.'}
        
    # Cari menu
    menu_mask = (df_final['menu_name'].str.lower() == menu_name.lower()) & (df_final['city_name'].str.lower() == city_name.lower())
    menu_data = df_final[menu_mask]
    
    if menu_data.empty:
        return {'error': f"Menu '{menu_name}' tidak ditemukan di kota '{city_name}'."}
        
    first_row = menu_data.iloc[0]
    
    # Kumpulkan daftar semua restoran (merchants) yang menjual menu tersebut
    merchants = []
    for _, row in menu_data.iterrows():
        merchants.append({
            'merchant_name': row['merchant_name'],
            'merchant_avg_rating': float(row.get('merchant_avg_rating', 0)),
            'merchant_review_count': int(row.get('merchant_review_count', 0)),
            'merchant_address': row.get('merchant_address', ''),
            'price_min': float(row.get('price_min', 0)),
            'price_max': float(row.get('price_max', 0)),
            'opening_time': str(row.get('opening_time', '')),
            'closing_time': str(row.get('closing_time', '')),
        })
        
    return {
        'menu_name': first_row['menu_name'],
        'menu_category': first_row['menu_category'],
        'menu_description': first_row['menu_description'],
        'menu_image_url': first_row.get('menu_image_url', ''),
        'flavor_tags': first_row['flavor_tags'],
        'city_name': first_row['city_name'],
        'country_name': first_row.get('country_name', ''),
        'currency_code': first_row.get('currency_code', ''),
        'currency_symbol': first_row.get('currency_symbol', ''),
        'merchants': merchants
    }

def get_all_menus():
    """Mengembalikan semua menu unik yang tersedia."""

    if df_unique_menu is None:
        return []

    menus = (
        df_unique_menu[
            [
                'menu_name',
                'menu_category',
                'city_name',
                'country_name'
            ]
        ]
        .drop_duplicates()
        .sort_values('menu_name')
    )

    return menus.to_dict(orient='records')


def search_menus(query):
    """Autocomplete pencarian menu."""

    if df_unique_menu is None:
        return []

    result = df_unique_menu[
        df_unique_menu['menu_name']
        .str.contains(query, case=False, na=False)
    ]

    result = (
        result[
            [
                'menu_name',
                'menu_category',
                'city_name',
                'country_name'
            ]
        ]
        .drop_duplicates()
        .sort_values('menu_name')
    )

    return result.to_dict(orient='records')


def get_all_merchants():
    """Mengembalikan semua merchant unik."""

    if df_final is None:
        return []

    merchants = (
        df_final[
            [
                'merchant_name',
                'city_name',
                'country_name'
            ]
        ]
        .drop_duplicates()
        .sort_values('merchant_name')
    )

    return merchants.to_dict(orient='records')

def get_available_countries():
    if df_unique_menu is None:
        return []

    countries = (
        df_unique_menu['country_name']
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return countries