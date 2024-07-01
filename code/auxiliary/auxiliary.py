import numpy as np

def transform_to_weight_table_primary_modeler(primary_modelers_scores, opinion_certainty_matrix, score_range_size, low_weight_for_unselected_features):
    primary_modelers_scores = np.atleast_2d(np.array(primary_modelers_scores))
    weight_table = np.full((primary_modelers_scores.shape[1], score_range_size, primary_modelers_scores.shape[0]), low_weight_for_unselected_features)
    
    for i in range(primary_modelers_scores.shape[0]):
        for j in range(primary_modelers_scores.shape[1]):
            if opinion_certainty_matrix[i, j] != 0 and primary_modelers_scores[i, j] != 0:
                score = primary_modelers_scores[i, j] - 1  # Adjust for 0-based indexing
                weight_table[i, score, j] += opinion_certainty_matrix[i, j]


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
        Samsung_expert_opinion_weight_tables[:, :, i] = transform_to_weight_table_primary_expert(expert_opinions[i], np.ones((1,3)), score_range_size, 0)

    Iphone_expert_opinion_weight_tables = np.zeros((3, score_range_size, 10))
    for i in range(10):
        Iphone_expert_opinion_weight_tables[:, :, i] = transform_to_weight_table_primary_expert(np.atleast_2d(expert_opinions[i]), np.ones(3), score_range_size, 0)

    Xiaomi_expert_opinion_weight_tables = np.zeros((3, score_range_size, 10))
    for i in range(20):
        Xiaomi_expert_opinion_weight_tables[:, :, i] = transform_to_weight_table_primary_expert(np.atleast_2d(expert_opinions[i]), np.ones(3), score_range_size, 0)

    Samsung_expert_opinion_weight_tables_trust = Samsung_expert_opinion_weight_tables * Trust_matrix[0, :, np.newaxis]
    Iphone_expert_opinion_weight_tables_trust = Iphone_expert_opinion_weight_tables * Trust_matrix[1, :, np.newaxis]
    Xiaomi_expert_opinion_weight_tables_trust = Xiaomi_expert_opinion_weight_tables * Trust_matrix[2, :, np.newaxis]

    The_cumulative_expert_weight = np.zeros((3, score_range_size, 3))
    The_cumulative_expert_weight[:, :, 0] = np.sum(Samsung_expert_opinion_weight_tables_trust, axis=2)
    The_cumulative_expert_weight[:, :, 1] = np.sum(Iphone_expert_opinion_weight_tables_trust, axis=2)
    The_cumulative_expert_weight[:, :, 2] = np.sum(Xiaomi_expert_opinion_weight_tables_trust, axis=2)

    The_primary_modelers_updated_weights = The_cumulative_expert_weight + The_primary_modelers_weights[:, :, np.newaxis]

    The_primary_modelers_opinion_features_old, The_primary_modelers_aposterior_old = primary_modelers_aposterior_B(The_primary_modelers_weights, P_I_B, score_preference)
    The_primary_modelers_opinion_features_updated, The_primary_modelers_aposterior_updated = primary_modelers_aposterior_B(The_primary_modelers_updated_weights, P_I_B, score_preference)

    return The_primary_modelers_aposterior_old, The_primary_modelers_aposterior_updated


# # Example data
# The_primary_modelers_scores = np.array([
#     [1, 1, 1],
#     [6, 5, 6],
#     [4, 4, 3]
# ])

# opinion_certainty = np.array([1, 0, 0])
# apply_certainty = True
# n_of_responders = np.array([10, 10, 10])
# Trust_matrix = np.ones((3, 10))
# score_preference = np.array([4, 4, 4])
# P_I_B = np.array([2, 2, 2])
# P_I_B = P_I_B / np.sum(P_I_B)
# score_range_size = 6
# low_weight_for_unselected_features = 0.01

# Samsung_e1 = [6, 6, 6]
# Samsung_e2 = [6, 6, 6]
# Samsung_e3 = [6, 6, 6]
# Samsung_e4 = [6, 6, 6]
# Samsung_e5 = [6, 6, 6]
# Samsung_e6 = [6, 6, 6]
# Samsung_e7 = [6, 6, 6]
# Samsung_e8 = [6, 6, 6]
# Samsung_e9 = [6, 6, 6]
# Samsung_e10 = [6, 6, 6]

# Iphone_e1 = [5, 5, 5]
# Iphone_e2 = [5, 6, 5]
# Iphone_e3 = [3, 4, 4]
# Iphone_e4 = [3, 4, 5]
# Iphone_e5 = [4, 5, 5]
# Iphone_e6 = [5, 6, 4]
# Iphone_e7 = [6, 6, 6]
# Iphone_e8 = [5, 6, 6]
# Iphone_e9 = [4, 3, 4]
# Iphone_e10 = [4, 6, 3]

# Xiaomi_e1 = [3, 4, 3]
# Xiaomi_e2 = [3, 3, 4]
# Xiaomi_e3 = [4, 3, 4]
# Xiaomi_e4 = [3, 3, 3]
# Xiaomi_e5 = [5, 4, 3]
# Xiaomi_e6 = [3, 5, 5]
# Xiaomi_e7 = [4, 5, 6]
# Xiaomi_e8 = [4, 3, 2]
# Xiaomi_e9 = [3, 4, 3]
# Xiaomi_e10 = [4, 3, 4]

# experts = (
#     Samsung_e1, Samsung_e2, Samsung_e3, Samsung_e4, Samsung_e5, 
#     Samsung_e6, Samsung_e7, Samsung_e8, Samsung_e9, Samsung_e10,
#     Iphone_e1, Iphone_e2, Iphone_e3, Iphone_e4, Iphone_e5, 
#     Iphone_e6, Iphone_e7, Iphone_e8, Iphone_e9, Iphone_e10,
#     Xiaomi_e1, Xiaomi_e2, Xiaomi_e3, Xiaomi_e4, Xiaomi_e5, 
#     Xiaomi_e6, Xiaomi_e7, Xiaomi_e8, Xiaomi_e9, Xiaomi_e10
# )

# # Running the simulation
# The_primary_modelers_aposterior_old, The_primary_modelers_aposterior_updated = simulated_example(
#     The_primary_modelers_scores, opinion_certainty, apply_certainty, n_of_responders, Trust_matrix, score_preference, P_I_B, score_range_size, low_weight_for_unselected_features, *experts
# )

# print("The Primary Modeler's Aposterior Old:", The_primary_modelers_aposterior_old)
# print("The Primary Modeler's Aposterior Updated:", The_primary_modelers_aposterior_updated)



