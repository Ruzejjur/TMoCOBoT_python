import numpy as np
import time

def generate_primary_modeler_weights(primary_modeler_scores, opinion_certainty_array, score_range, initial_feature_weight):
    primary_modeler_scores = np.atleast_2d(np.array(primary_modeler_scores))
    weight_table = np.full((primary_modeler_scores.shape[1], score_range, primary_modeler_scores.shape[0]), initial_feature_weight)
    
    # Create boolean masks
    non_zero_certainty = opinion_certainty_array != 0
    non_zero_scores = primary_modeler_scores != 0

    # Combine masks
    mask = non_zero_certainty[:, np.newaxis] & non_zero_scores

    # Indices where both conditions are met
    i_indices, j_indices = np.where(mask)

    # Adjust scores
    scores = primary_modeler_scores[mask] - 1  # Adjust for 0-based indexing

    # Update weight_table
    weight_table[j_indices, scores, i_indices] += opinion_certainty_array[i_indices]

    return weight_table

def generate_expert_weights(expert_scores, score_range, initial_feature_weight):
    expert_scores = np.atleast_2d(np.array(expert_scores))
    weight_table = np.full((expert_scores.shape[1], score_range, expert_scores.shape[0]), initial_feature_weight)
    
    non_zero_scores = expert_scores != 0
    
    j_indices = np.where(expert_scores[non_zero_scores])
    
    scores = expert_scores[non_zero_scores] - 1 # Adjust for zero based indexing
    
    weight_table[j_indices, scores, 0] += 1

    return weight_table

def compute_primary_modeler_posterior_brands(primary_modeler_weights, primary_modeler_brand_pref, primary_modeler_score_preference):
    primary_modeler_posterior_brands = np.ones((primary_modeler_weights.shape[2],))
    primary_modeler_opinion_features = np.zeros_like(primary_modeler_weights)

    for i in range(primary_modeler_weights.shape[2]):
        for j in range(primary_modeler_weights.shape[0]):
            primary_modeler_opinion_features[j, :, i] = primary_modeler_weights[j, :, i] / np.sum(primary_modeler_weights[j, :, i])

    primary_modeler_preference_of_scores = primary_modeler_opinion_features.copy()
    for i in range(primary_modeler_preference_of_scores.shape[0]):
        primary_modeler_preference_of_scores[i, :primary_modeler_score_preference[i]-1, :] = 0

    max_probabilities_in_preference_matrix = np.max(primary_modeler_preference_of_scores, axis=1)

    for i in range(primary_modeler_weights.shape[2]):
        primary_modeler_posterior_brands[i] = np.prod(max_probabilities_in_preference_matrix[:, i]) * primary_modeler_brand_pref[i]

    primary_modeler_posterior_brands = primary_modeler_posterior_brands / np.sum(primary_modeler_posterior_brands)
    
    return primary_modeler_opinion_features, primary_modeler_posterior_brands


def simulated_example(primary_modeler_scores, opinion_certainty_array, apply_certainty, number_of_responders,
                      trust_matrix, score_preference, primary_modeler_brand_pref, score_range, initial_feature_weight, *expert_opinions):
    if apply_certainty:
        # Solving the issue of primary modelers opinion being diminished
        opinion_certainty_array = opinion_certainty_array * number_of_responders
    else:
        # Make sure that the opinion certainty array is ones (equvivalent to no certainty being applied)
        opinion_certainty_array = np.ones_like(opinion_certainty_array)

    # Generating primary modelers weights
    primary_modeler_weights = generate_primary_modeler_weights(primary_modeler_scores, opinion_certainty_array, score_range, initial_feature_weight)

    # Generate expert opinion weights for samsung experts
    # Note 1: Watch out for data type, needs to be float otherwise trust is not applied correctly
    # TODO: In the future, look into more efficient container, as the float 32 array would take up to much space
    # TODO: Fully vectorise this operation
    samsung_expert_opinion_weights = np.array([generate_expert_weights(expert_opinions[i], score_range, 0) for i in range(10)], dtype=np.float32)

    samsung_expert_opinion_weights = np.squeeze(samsung_expert_opinion_weights, axis=-1)
    samsung_expert_opinion_weights = np.transpose(samsung_expert_opinion_weights, (1,2,0))

    # Generate expert opinion weights for iphone experts
    # Note 1: Watch out for data type, needs to be float otherwise trust is not applied correctly
    # TODO: In the future, look into more efficient container, as the float 32 array would take up to much space
    # TODO: Fully vectorise this operation
    iphone_expert_opinion_weights = np.array([generate_expert_weights(expert_opinions[i], score_range, 0) for i in range(10,20)], dtype=np.float32)

    iphone_expert_opinion_weights = np.squeeze(iphone_expert_opinion_weights, axis=-1)
    iphone_expert_opinion_weights = np.transpose(iphone_expert_opinion_weights, (1,2,0))
                
    # Generate expert opinion weights for iphone experts
    # Note 1: Watch out for data type, needs to be float otherwise trust is not applied correctly
    # TODO: In the future, look into more efficient container, as the float 32 array would take up to much space
    # TODO: Fully vectorise this operation
    xiaomi_expert_opinion_weights = np.array([generate_expert_weights(expert_opinions[i], score_range, 0) for i in range(20,30)], dtype=np.float32)

    xiaomi_expert_opinion_weights = np.squeeze(xiaomi_expert_opinion_weights, axis=-1)
    xiaomi_expert_opinion_weights = np.transpose(xiaomi_expert_opinion_weights, (1,2,0))

    # TODO: Replace the shape call with number of experts input into the function
    i_indices =  np.arange(samsung_expert_opinion_weights.shape[2])
    samsung_expert_opinion_weights_trust = samsung_expert_opinion_weights[:,:,i_indices] * trust_matrix[0,i_indices]
   
    # TODO: Replace the shape call with number of experts input into the function  
    i_indices =  np.arange(iphone_expert_opinion_weights.shape[2])
    iphone_expert_opinion_weights_trust = iphone_expert_opinion_weights[:,:,i_indices] * trust_matrix[1,i_indices]

    # TODO: Replace the shape call with number of experts input into the function 
    i_indices =  np.arange(xiaomi_expert_opinion_weights.shape[2])
    xiaomi_expert_opinion_weights_trust = xiaomi_expert_opinion_weights[:,:,i_indices] * trust_matrix[2,i_indices]
    
    
    cumulative_expert_weights = np.zeros((3, score_range, 3))
    cumulative_expert_weights[:, :, 0] = np.sum(samsung_expert_opinion_weights_trust, axis=2)
    cumulative_expert_weights[:, :, 1] = np.sum(iphone_expert_opinion_weights_trust, axis=2)
    cumulative_expert_weights[:, :, 2] = np.sum(xiaomi_expert_opinion_weights_trust, axis=2)

    primary_modeler_updated_weights = cumulative_expert_weights + primary_modeler_weights

    _ , primary_modeler_posterior_old = compute_primary_modeler_posterior_brands(primary_modeler_weights, primary_modeler_brand_pref, score_preference)
    _ , primary_modeler_posterior_updated = compute_primary_modeler_posterior_brands(primary_modeler_updated_weights, primary_modeler_brand_pref, score_preference)

    return primary_modeler_posterior_old, primary_modeler_posterior_updated
