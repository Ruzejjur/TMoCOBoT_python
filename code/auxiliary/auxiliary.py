import numpy as np

def transform_to_weight_table_primary_modeler(primary_modelers_scores, opinion_certainty_matrix, score_range_size, low_weight_for_unselected_features):
    primary_modelers_scores = np.atleast_2d(np.array(primary_modelers_scores))
    weight_table = np.full((primary_modelers_scores.shape[1], score_range_size, primary_modelers_scores.shape[0]), low_weight_for_unselected_features)
    
    for i in range(primary_modelers_scores.shape[0]):
        for j in range(primary_modelers_scores.shape[1]):
            if opinion_certainty_matrix[i, j] != 0 and primary_modelers_scores[i, j] != 0:
                score = primary_modelers_scores[i, j] - 1  # Adjust for 0-based indexing
                weight_table[j, score, i] += opinion_certainty_matrix[i, j]


    return weight_table

def transform_to_weight_table_primary_expert(primary_modelers_scores, opinion_certainty_matrix, score_range_size, low_weight_for_unselected_features):
    primary_modelers_scores = np.atleast_2d(np.array(primary_modelers_scores))
    weight_table = np.full((primary_modelers_scores.shape[1], score_range_size, primary_modelers_scores.shape[0]), low_weight_for_unselected_features)
    
    for i in range(primary_modelers_scores.shape[0]):
        for j in range(primary_modelers_scores.shape[1]):
            if opinion_certainty_matrix[i, j] != 0 and primary_modelers_scores[i, j] != 0:
                score = primary_modelers_scores[i, j] - 1  # Adjust for 0-based indexing
                weight_table[j, score, 0] += opinion_certainty_matrix[i, j]


    return weight_table

def primary_modelers_aposterior_B(primary_modelers_weights, P_I_B, primary_modelers_score_preference):
    primary_modelers_aposterior_B = np.ones((primary_modelers_weights.shape[2],))
    primary_modelers_opinion_features = np.zeros_like(primary_modelers_weights)

    for i in range(primary_modelers_weights.shape[2]):
        for j in range(primary_modelers_weights.shape[0]):
            primary_modelers_opinion_features[j, :, i] = primary_modelers_weights[j, :, i] / np.sum(primary_modelers_weights[j, :, i])

    primary_modelers_preference_of_scores = primary_modelers_opinion_features.copy()
    for i in range(primary_modelers_preference_of_scores.shape[0]):
        primary_modelers_preference_of_scores[i, :primary_modelers_score_preference[i]-1, :] = 0

    max_probabilities_in_preference_matrix = np.max(primary_modelers_preference_of_scores, axis=1)

    for i in range(primary_modelers_weights.shape[2]):
        primary_modelers_aposterior_B[i] = np.prod(max_probabilities_in_preference_matrix[:, i]) * P_I_B[i]

    primary_modelers_aposterior_B = primary_modelers_aposterior_B / np.sum(primary_modelers_aposterior_B)
    
    return primary_modelers_opinion_features, primary_modelers_aposterior_B


def simulated_example(The_primary_modelers_scores, opinion_certainty, apply_certainty, n_of_responders, Trust_matrix, score_preference, P_I_B, score_range_size, low_weight_for_unselected_features, *expert_opinions):
    opinion_certainty_matrix = np.ones((3, 3))
    
    if apply_certainty:
        for i in range(3):
            opinion_certainty_matrix[i, :] = opinion_certainty[i] * n_of_responders[i]
    else:
        opinion_certainty_matrix = np.ones_like(opinion_certainty_matrix)

    The_primary_modelers_weights = transform_to_weight_table_primary_modeler(The_primary_modelers_scores, opinion_certainty_matrix, score_range_size, low_weight_for_unselected_features)

    Samsung_expert_opinion_weight_tables = np.zeros((3, score_range_size, 10))
    for i in range(10):
        expert_opinion_weight_table = transform_to_weight_table_primary_expert(expert_opinions[i], np.ones((1,3)), score_range_size, 0)
        for j in range(3):
            for k in range(6):
                Samsung_expert_opinion_weight_tables[j, k, i] = expert_opinion_weight_table[j,k]
    
    Iphone_expert_opinion_weight_tables = np.zeros((3, score_range_size, 10))
    for i in range(10,20):
        expert_opinion_weight_table = transform_to_weight_table_primary_expert(expert_opinions[i], np.ones((1,3)), score_range_size, 0)
        for j in range(3):
            for k in range(6):
                Iphone_expert_opinion_weight_tables[j, k, i-10] = expert_opinion_weight_table[j,k]

    Xiaomi_expert_opinion_weight_tables = np.zeros((3, score_range_size, 10))
    for i in range(20,30):
        expert_opinion_weight_table = transform_to_weight_table_primary_expert(expert_opinions[i], np.ones((1,3)), score_range_size, 0)
        for j in range(3):
            for k in range(6):
                Xiaomi_expert_opinion_weight_tables[j, k, i-20] = expert_opinion_weight_table[j,k]

    Samsung_expert_opinion_weight_tables_trust = np.zeros_like(Samsung_expert_opinion_weight_tables)
    for i in range(Samsung_expert_opinion_weight_tables.shape[2]):
        Samsung_expert_opinion_weight_tables_trust[:,:,i] = Samsung_expert_opinion_weight_tables[:,:,i]*Trust_matrix[0,i]
    
    Iphone_expert_opinion_weight_tables_trust = np.zeros_like(Iphone_expert_opinion_weight_tables)
    for i in range(Iphone_expert_opinion_weight_tables.shape[2]):
        Iphone_expert_opinion_weight_tables_trust[:,:,i] = Iphone_expert_opinion_weight_tables[:,:,i]*Trust_matrix[1,i]
    
    Xiaomi_expert_opinion_weight_tables_trust = np.zeros_like(Xiaomi_expert_opinion_weight_tables)
    for i in range(Xiaomi_expert_opinion_weight_tables.shape[2]):
        Xiaomi_expert_opinion_weight_tables_trust[:,:,i] = Xiaomi_expert_opinion_weight_tables[:,:,i]*Trust_matrix[2,i]

    The_cumulative_expert_weight = np.zeros((3, score_range_size, 3))
    The_cumulative_expert_weight[:, :, 0] = np.sum(Samsung_expert_opinion_weight_tables_trust, axis=2)
    The_cumulative_expert_weight[:, :, 1] = np.sum(Iphone_expert_opinion_weight_tables_trust, axis=2)
    The_cumulative_expert_weight[:, :, 2] = np.sum(Xiaomi_expert_opinion_weight_tables_trust, axis=2)

    The_primary_modelers_updated_weights = The_cumulative_expert_weight + The_primary_modelers_weights

    The_primary_modelers_opinion_features_old, The_primary_modelers_aposterior_old = primary_modelers_aposterior_B(The_primary_modelers_weights, P_I_B, score_preference)
    The_primary_modelers_opinion_features_updated, The_primary_modelers_aposterior_updated = primary_modelers_aposterior_B(The_primary_modelers_updated_weights, P_I_B, score_preference)

    return The_primary_modelers_aposterior_old, The_primary_modelers_aposterior_updated
