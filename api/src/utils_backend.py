import numpy as np

def get_items_rated_by_user(rate_matrix,user_id):
    y_user=rate_matrix[:,0]
    ids=np.where(y_user==user_id)[0]
    items_id=rate_matrix[ids,1]
    scores=rate_matrix[ids,2]
    return (items_id,scores)