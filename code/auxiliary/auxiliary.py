import numpy as np

def generate_primary_modeler_weights(primary_modeler_scores, opinion_certainty_matrix, score_range, initial_feature_weight):
    primary_modeler_scores = np.atleast_2d(np.array(primary_modeler_scores))
    weight_table = np.full((primary_modeler_scores.shape[1], score_range, primary_modeler_scores.shape[0]), initial_feature_weight)
    
    for i in range(primary_modeler_scores.shape[0]):
        for j in range(primary_modeler_scores.shape[1]):
            if opinion_certainty_matrix[i, j] != 0 and primary_modeler_scores[i, j] != 0:
                score = primary_modeler_scores[i, j] - 1  # Adjust for 0-based indexing
                weight_table[j, score, i] += opinion_certainty_matrix[i, j]


    return weight_table

def generate_expert_weights(expert_scores, opinion_certainty_matrix, score_range, initial_feature_weight):
    expert_scores = np.atleast_2d(np.array(expert_scores))
    weight_table = np.full((expert_scores.shape[1], score_range, expert_scores.shape[0]), initial_feature_weight)
    
    for i in range(expert_scores.shape[0]):
        for j in range(expert_scores.shape[1]):
            if opinion_certainty_matrix[i, j] != 0 and expert_scores[i, j] != 0:
                score = expert_scores[i, j] - 1  # Adjust for 0-based indexing
                weight_table[j, score, 0] += opinion_certainty_matrix[i, j]


    return weight_table

def primary_modelers_posterior_brands(primary_modelers_weights, P_I_B, primary_modelers_score_preference):
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


def simulated_example(primary_modeler_scores, opinion_certainty_matrix, apply_certainty, number_of_responders, trust_matrix, score_preference, primary_modeler_brand_pref, score_range, initial_feature_weight, *expert_opinions):
    opinion_certainty_matrix = np.ones((3, 3))
    
    if apply_certainty:
        for i in range(3):
            opinion_certainty_matrix[i, :] = opinion_certainty_matrix[i] * number_of_responders[i]
    else:
        opinion_certainty_matrix = np.ones_like(opinion_certainty_matrix)

    primary_modeler_weights = generate_primary_modeler_weights(primary_modeler_scores, opinion_certainty_matrix, score_range, initial_feature_weight)

    samsung_expert_opinion_weights = np.zeros((3, score_range, 10))
    for i in range(10):
        expert_opinion_weight_table = generate_expert_weights(expert_opinions[i], np.ones((1,3)), score_range, 0)
        for j in range(3):
            for k in range(6):
                samsung_expert_opinion_weights[j, k, i] = expert_opinion_weight_table[j,k]
    
    iphone_expert_opinion_weights = np.zeros((3, score_range, 10))
    for i in range(10,20):
        expert_opinion_weight_table = generate_expert_weights(expert_opinions[i], np.ones((1,3)), score_range, 0)
        for j in range(3):
            for k in range(6):
                iphone_expert_opinion_weights[j, k, i-10] = expert_opinion_weight_table[j,k]

    xiaomi_expert_opinion_weights = np.zeros((3, score_range, 10))
    for i in range(20,30):
        expert_opinion_weight_table = generate_expert_weights(expert_opinions[i], np.ones((1,3)), score_range, 0)
        for j in range(3):
            for k in range(6):
                xiaomi_expert_opinion_weights[j, k, i-20] = expert_opinion_weight_table[j,k]

    samsung_expert_opinion_weights_trust = np.zeros_like(samsung_expert_opinion_weights)
    for i in range(samsung_expert_opinion_weights.shape[2]):
        samsung_expert_opinion_weights_trust[:,:,i] = samsung_expert_opinion_weights[:,:,i]*trust_matrix[0,i]
    
    iphone_expert_opinion_weights_trust = np.zeros_like(iphone_expert_opinion_weights)
    for i in range(iphone_expert_opinion_weights.shape[2]):
        iphone_expert_opinion_weights_trust[:,:,i] = iphone_expert_opinion_weights[:,:,i]*trust_matrix[1,i]
    
    xiaomi_expert_opinion_weights_trust = np.zeros_like(xiaomi_expert_opinion_weights)
    for i in range(xiaomi_expert_opinion_weights.shape[2]):
        xiaomi_expert_opinion_weights_trust[:,:,i] = xiaomi_expert_opinion_weights[:,:,i]*trust_matrix[2,i]

    cumulative_expert_weights = np.zeros((3, score_range, 3))
    cumulative_expert_weights[:, :, 0] = np.sum(samsung_expert_opinion_weights_trust, axis=2)
    cumulative_expert_weights[:, :, 1] = np.sum(iphone_expert_opinion_weights_trust, axis=2)
    cumulative_expert_weights[:, :, 2] = np.sum(xiaomi_expert_opinion_weights_trust, axis=2)

    primary_modeler_updated_weights = cumulative_expert_weights + primary_modeler_weights

    primary_modeler_opinion_old, primary_modeler_posterior_old = primary_modelers_posterior_brands(primary_modeler_weights, primary_modeler_brand_pref, score_preference)
    primary_modeler_opinion_updated, primary_modeler_posterior_updated = primary_modelers_posterior_brands(primary_modeler_updated_weights, primary_modeler_brand_pref, score_preference)

    return primary_modeler_posterior_old, primary_modeler_posterior_updated
