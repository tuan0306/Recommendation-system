import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import sys
sys.path.append('..')
from src.utils import get_items_rated_by_user

class ContentBasedFiltering:
    def __init__(self,X_tfidf,n_users,_lambda=1.0):
        self.X_tfidf=X_tfidf
        self.n_users=n_users
        self._lambda=_lambda
        self.n_features=X_tfidf.shape[1]
        self.W=np.zeros((self.n_features,self.n_users))
        self.b=np.zeros((1,self.n_users))
        self.Yhat=None
        self.items_sim_matrix=None
    def fit(self,train_data):
        for n in range(self.n_users):
            ids,scores=get_items_rated_by_user(train_data,n)
            model=Ridge(alpha=self._lambda,fit_intercept=True)
            X_user_rated=self.X_tfidf[ids,:]
            model.fit(X_user_rated,scores)
            self.W[:,n]=model.coef_
            self.b[0,n]=model.intercept_
        self.Yhat=self.X_tfidf.dot(self.W)+self.b
        self.items_sim_matrix=cosine_similarity(self.X_tfidf,self.X_tfidf)
    def predict(self,test_data):
        predict_all=[]
        for n in range(self.n_users):
            ids,scores=get_items_rated_by_user(test_data,n)
            score_predict=self.Yhat[ids,n]
            predict_all.append(score_predict)
        return np.array(predict_all)
    def predict_one_user(self,user_id,test_data):
        ids,scores=get_items_rated_by_user(test_data,user_id)
        movies_id=(ids+1).tolist()
        predict_ratings=self.Yhat[ids,user_id].tolist()
        real_ratings=scores.tolist()
        return movies_id,real_ratings,predict_ratings
    def predict_one_user_item(self,user_id,item_id,test_data):
        ids,scores=get_items_rated_by_user(test_data,user_id)
        id_movie_index=np.where(ids==item_id)[0]
        return scores[id_movie_index][0],self.Yhat[item_id,user_id]
    def recommend_similar_items(self, item_id, top=10):
        sim_scores=self.items_sim_matrix[item_id]
        recommend_items=np.argsort(sim_scores)[-(top+1):-1][::-1]
        return recommend_items
    def RMSE(self,rates_data):
        y_real=rates_data[:,2]
        y_pre=[]
        for n in range(self.n_users):
            ids,real_scores=get_items_rated_by_user(rates_data,n)
            predict_scores=self.Yhat[ids,n]
            y_pre.extend(predict_scores)
        return np.sqrt(mean_squared_error(y_real,y_pre))


class NBCF:
    def __init__(self,n_users,n_items,k,dist_func=cosine_similarity,uuCF=1):
        self.uuCF=uuCF
        self.dist_func=dist_func
        self.Ybar=None
        self.k=k
        self.n_users=n_users if uuCF else n_items
        self.n_items=n_items if uuCF else n_users
    def normalize(self,data):
        self.Y_data=data if self.uuCF else data[:,[1,0,2]]
        self.Ybar=self.Y_data.copy()
        users=self.Y_data[:,0]
        self.mu=np.zeros((self.n_users,))
        for i in range(self.n_users):
            ids=np.where(users==i)[0].astype(int)
            ratings=self.Y_data[ids,2]
            m=np.mean(ratings)
            if np.isnan(m):
                m=0
            self.mu[i]=m
            self.Ybar[ids,2]=ratings-m
        self.Ybar=sparse.coo_matrix((self.Ybar[:,2],(self.Ybar[:,1],self.Ybar[:,0])),
                                    (self.n_items,self.n_users))
        self.Ybar=self.Ybar.tocsr()
        
    def similarity(self):
        self.S=self.dist_func(self.Ybar.T,self.Ybar.T)
    
    def fit(self,data):
        self.normalize(data)
        self.similarity()
    
    def _predict(self,u,i,normalized=1):
        ids=np.where(self.Y_data[:,1]==i)[0].astype(np.int32)
        users_rated_i=(self.Y_data[ids,0]).astype(np.int32)
        sim=self.S[u,users_rated_i]
        a=np.argsort(sim)[-self.k:]
        nearest=sim[a]
        r=self.Ybar[i,users_rated_i[a]]
        if normalized:
            return (r*nearest)[0]/(np.abs(nearest).sum()+1e-9)
        return (r*nearest)[0]/(np.abs(nearest).sum()+1e-9)+self.mu[u]
    
    def predict(self,u,i,normalized=1):
        if self.uuCF: return self._predict(u,i,normalized)
        return self._predict(i,u,normalized)
    
    def predict_test(self,ratings_test):
        n_tests=ratings_test.shape[0]
        prediction=np.zeros((n_tests,))
        for n in range(n_tests):
            prediction[n]=self.predict(ratings_test[n,0],ratings_test[n,1],normalized=0)
        return prediction
    
    def RMSE(self,data,predict):
        return np.sqrt(mean_squared_error(data[:,2],predict))
    
    def recommend_for_user(self,u,top=10):
        if self.uuCF:
            ids=np.where(self.Y_data[:,0]==u)[0].astype(np.int32)
            items_rated_by_u=self.Y_data[ids,1].tolist()
            n=self.n_items
        else:
            ids=np.where(self.Y_data[:,1]==u)[0].astype(np.int32)
            items_rated_by_u=self.Y_data[ids,0].tolist()
            n=self.n_users
        a=np.zeros((n,))
        a[items_rated_by_u]=-1e9
        for i in range(n):
            if i not in items_rated_by_u:
                a[i]=self.predict(u,i)
        recommended_items=np.argsort(a)[-top:][::-1]
        return recommended_items
    
    def recommend_simiar_items(self,i,top=10):
        if self.uuCF == 1:
            raise ValueError("Method only available for Item-Item CF (uuCF=0).")
        sim_scores=self.S[i]
        similar_items=np.argsort(sim_scores)[-(top+1):-1][::-1]
        return similar_items
            