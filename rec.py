# import firebase_admin
# from firebase_admin import credentials, firestore
# from sklearn.neighbors import NearestNeighbors
# from dotenv import load_dotenv
# from flask import Flask, jsonify

# # Load environment variables
# load_dotenv()

# # Get the service key path from environment variable
# service_key_path = os.getenv('FIREBASE_SERVICE_KEY_PATH')

# if service_key_path is None:
#     raise ValueError("The environment variable FIREBASE_SERVICE_KEY_PATH is not set.")

# # Initialize Firebase
# cred = credentials.Certificate(service_key_path)
# firebase_admin.initialize_app(cred)

# db = firestore.client()

# def collection_recommender_engine(collection_name, matrix, cf_model, n_recs):
#     cf_model.fit(matrix)
    
#     collection_index = process.extractOne(collection_name, matrix.index)[2]
    
#     distances, indices = cf_model.kneighbors(matrix.iloc[collection_index].values.reshape(1, -1), n_neighbors=n_recs)
#     collection_rec_ids = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    
#     cf_recs = []
#     for i in collection_rec_ids:
#         cf_recs.append({'Title': matrix.index[i[0]], 'Distance': i[1]})
    
#     df = pd.DataFrame(cf_recs, index=range(1, n_recs))
     
#     return df

# collections = []
# collection_docs = db.collection('collections').stream()
# for doc in collection_docs:
#     collections.append(doc.document('data').get('title')) 

# collection_titles = collections
# num_collections = len(collection_titles)
# dummy_relevancy_matrix = np.random.rand(num_collections, num_collections)

# dummy_relevancy_matrix = (dummy_relevancy_matrix + dummy_relevancy_matrix.T) / 2

# np.fill_diagonal(dummy_relevancy_matrix, 1)

# relevancy_df = pd.DataFrame(dummy_relevancy_matrix, index=collection_titles, columns=collection_titles)
# relevancy_df.head()

# cf_knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
# cf_knn_model.fit(relevancy_df)

# n_recs = 10
# print(collection_recommender_engine('Some Collection Title', relevancy_df, cf_knn_model, n_recs))



