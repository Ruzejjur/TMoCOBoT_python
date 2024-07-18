import numpy as np
import time

def generate_primary_modeler_weights(primary_modeler_scores, opinion_certainty_array, score_range, initial_feature_weight):
    """
    Generates weights for the primary modeler's scores, adjusted by opinion certainty.

    Parameters:
    primary_modeler_scores (np.ndarray): Scores given by the primary modeler for different brands and features.
    opinion_certainty_array (np.ndarray): Array representing the certainty of the primary modeler's opinions.
    score_range (int): Highest possible score for each feature.
    initial_feature_weight (float): Initial feature weight for the Dirichlet distribution.

    Returns:
    np.ndarray: A weight table adjusted by the primary modeler's scores and opinion certainty.
    """

    # Initialize a weight table filled with the initial feature weight
    # Shape: (number of features, score_range, number of brands)
    weight_table = np.full((primary_modeler_scores.shape[1], score_range, primary_modeler_scores.shape[0]), initial_feature_weight, dtype=np.float32)
    
    # Create boolean masks to identify non-zero certainty and non-zero scores
    non_zero_certainty = opinion_certainty_array != 0
    non_zero_scores = primary_modeler_scores != 0

    # Combine masks to identify positions where both conditions are met
    mask = non_zero_certainty[:, np.newaxis] & non_zero_scores

    # Find indices where the combined mask is true
    i_indices, j_indices = np.where(mask)

    # Adjust scores to be 0-based (subtract 1 from each score)
    scores = primary_modeler_scores[mask] - 1

    # Update the weight table with the adjusted scores and opinion certainty
    weight_table[j_indices, scores, i_indices] += opinion_certainty_array[i_indices]

    return weight_table


def generate_expert_weights(expert_scores, score_range, initial_feature_weight):
    """
    Generates weights for expert scores, adjusting by a given initial feature weight.

    Parameters:
    expert_scores (np.ndarray): Scores given by experts for different brands and features.
    score_range (int): Highest possible score for each feature.
    initial_feature_weight (float): Initial feature weight for the Dirichlet distribution.

    Returns:
    np.ndarray: A weight table adjusted by the expert scores.
    """

    # Initialize a weight table filled with the initial feature weight
    # Shape: (number of features, score_range, number of experts)
    weight_table = np.full((expert_scores.shape[1], score_range, expert_scores.shape[0]), initial_feature_weight, dtype=np.int8)
    
    # Create a boolean mask to identify non-zero scores
    non_zero_scores = expert_scores != 0

    # Find indices where the scores are non-zero
    i_indices, j_indices = np.nonzero(non_zero_scores)
    
    # Adjust scores to be 0-based (subtract 1 from each score)
    scores = expert_scores[non_zero_scores] - 1
    
    # Update the weight table with the adjusted scores
    # Add 1 to the weight table at the positions indicated by the indices
    weight_table[j_indices, scores, i_indices] += 1

    return weight_table


def compute_primary_modeler_posterior_brands(primary_modeler_weights, primary_modeler_brand_pref, primary_modeler_score_preference):
    """
    Computes the primary modeler's posterior probabilities for brands based on primary modeler's scores in form of weights, brand preferences, and score preferences.

    Parameters:
    primary_modeler_weights (np.ndarray): Weights for each feature, score, and brand.
    primary_modeler_brand_pref (np.ndarray): Distribution representing primary modeler's brand preference.
    primary_modeler_score_preference (np.ndarray): Primary modeler's score preference for each feature.

    Returns:
    tuple: A tuple containing:
        - primary_modeler_opinion_features (np.ndarray): Normalized opinion on features of each brand for the primary modeler.
        - primary_modeler_posterior_brands (np.ndarray): Posterior probabilities for each brand.
    """

    # Initialize posterior probabilities for brands as np.ndarray of ones
    primary_modeler_posterior_brands = np.ones((primary_modeler_weights.shape[2],), dtype=np.float16)
    
    # Initialize opinion features for the primary modeler
    primary_modeler_opinion_features = np.zeros_like(primary_modeler_weights, dtype=np.float16)

    # Calculate the sum along the second dimension (axis=1) and keep the dimensions for broadcasting
    sums = np.sum(primary_modeler_weights, axis=1, keepdims=True)

    # Perform the division using broadcasting to normalize the weights
    primary_modeler_opinion_features = primary_modeler_weights / sums

    ## * Filter scores based on preferences
    # * Testing for speed showed that vector implementation is slower than for loop, but might be faster with large datasets
    
    # Create a different reference to primary_modeler_opinion_features for better readibility
    primary_modeler_preferred_scores = primary_modeler_opinion_features
    
    # Generate an array of row indices corresponding to the scores
    # * primary_modeler_preferred_scores.shape[1] gives the number of possible scores (score range)
    # * np.arange creates an array of integers from 0 to (score range - 1)
    # * reshape(1, -1) changes the shape to (1, score range) for broadcasting purposes
    rows = np.arange(primary_modeler_preferred_scores.shape[1], dtype=np.int8).reshape(1, -1)

    # Create a mask based on the primary modeler's score preferences
    # * primary_modeler_score_preference is an array of preferred scores for each feature
    # * Subtract 1 to adjust for 0-based indexing (e.g., preference of 1 means the score should be >= 0 in 0-based indexing)
    # * Reshape the preferences to (number of features, 1) to align for broadcasting
    # * rows[:, None] expands the dimensions to (1, score range, 1)
    # * The comparison checks if each score (rows) is less than the preferred score minus 1
    mask = rows[:, None] < (primary_modeler_score_preference - 1)[:, None].reshape(-1, 1)

    # Transpose the mask to align dimensions correctly
    # * The mask has shape (number of features, score range, 1) after the previous step
    # * Transpose to (score range, number of features, 1)
    mask = mask.transpose(1, 2, 0)

    # Broadcast the mask to match the shape of primary_modeler_preferred_scores
    # * The original mask shape is (score range, number of features, 1)
    # * Expand to (number of brands, score range, number of features)
    # * primary_modeler_preferred_scores has shape (number of brands, score range, number of features)
    mask = np.broadcast_to(mask, (primary_modeler_preferred_scores.shape[0], primary_modeler_preferred_scores.shape[1], primary_modeler_preferred_scores.shape[2]))
    
    # Apply the mask to set the scores below preference to zero
    primary_modeler_preferred_scores[mask] = 0

    # Calculate the maximum probabilities for preferred scores
    max_probabilities_in_preference_matrix = np.max(primary_modeler_preferred_scores, axis=1)
    
    # Calculate the product of maximum probabilities across all features
    max_probabilities_in_preference_matrix_product = np.prod(max_probabilities_in_preference_matrix, axis=0)
    
    # Update the posterior probabilities for brands
    primary_modeler_posterior_brands = max_probabilities_in_preference_matrix_product * primary_modeler_brand_pref
    
    # Normalize the posterior probabilities to sum to 1
    primary_modeler_posterior_brands = primary_modeler_posterior_brands / np.sum(primary_modeler_posterior_brands)
    
    return primary_modeler_opinion_features, primary_modeler_posterior_brands



def simulated_example(primary_modeler_scores, opinion_certainty_array, apply_certainty, number_of_responders,
                      trust_matrix, primary_modeler_score_preference, primary_modeler_brand_pref, score_range, initial_feature_weight, 
                      samsung_expert_opinions, iphone_expert_opinions, xiaomi_expert_opinions):
    """
    Simulates the primary modeler's posterior preferences by combining initial scores with expert opinions.

    Parameters:
    primary_modeler_scores (np.ndarray): Scores given by the primary modeler for different brands and features.
    opinion_certainty_array (np.ndarray): Array representing the certainty of the primary modeler's opinions.
    apply_certainty (bool): Flag indicating whether to apply the certainty weights.
    number_of_responders (np.ndarray): Number of responders for each brand.
    trust_matrix (np.ndarray): Matrix representing the trust levels for each expert's opinion. Possible values: interval [0,1].
    score_preference (np.ndarray): Primary modeler's score preference for each feature.
    primary_modeler_brand_pref (np.ndarray): Distribution representing primary modeler's brand preference.
    score_range (int): Highest possible score for each feature.
    initial_feature_weight (float): Initial feature weight for the Dirichlet distribution.
    samsung_expert_opinions (np.ndarray): Expert opinions for Samsung.
    iphone_expert_opinions (np.ndarray): Expert opinions for iPhone.
    xiaomi_expert_opinions (np.ndarray): Expert opinions for Xiaomi.

    Returns:
    tuple: A tuple containing:
        - primary_modeler_posterior_initial (np.ndarray): Initial posterior probabilities for the primary modeler.
        - primary_modeler_posterior_updated (np.ndarray): Updated posterior probabilities for the primary modeler after considering expert opinions.
    """

    if apply_certainty:
        # Apply certainty by scaling the opinion certainty array with the number of responders
        opinion_certainty_array = opinion_certainty_array * number_of_responders
    else:
        # Ensure that the opinion certainty array is ones (equivalent to no certainty being applied)
        opinion_certainty_array = np.ones_like(opinion_certainty_array, dtype=np.float16)

    # Generate primary modeler's weights
    primary_modeler_weights = generate_primary_modeler_weights(primary_modeler_scores, opinion_certainty_array, score_range, initial_feature_weight)

    # Generate expert opinion weights for Samsung experts
    # ! Note 1: Ensure data type is float to correctly apply trust
    samsung_expert_opinion_weights = generate_expert_weights(samsung_expert_opinions, score_range, 0)

    # Generate expert opinion weights for iPhone experts
    # ! Note 1: Ensure data type is float to correctly apply trust
    iphone_expert_opinion_weights = generate_expert_weights(iphone_expert_opinions, score_range, 0)
                
    # Generate expert opinion weights for Xiaomi experts
    # ! Note 1: Ensure data type is float to correctly apply trust
    xiaomi_expert_opinion_weights = generate_expert_weights(xiaomi_expert_opinions, score_range, 0)

    ## * Apply trust matrix to expert opinion weights for Samsung

    # Create an array of expert indices for Samsung
    # This is essentially a range of numbers from 0 to the number of experts - 1
    i_indices = np.arange(samsung_expert_opinion_weights.shape[2], dtype=np.int32)

    # Multiply the expert opinion weights by the corresponding trust values for Samsung
    # * Note 1: samsung_expert_opinion_weights is a 3D array where the third dimension represents different experts
    # * trust_matrix[0, i_indices] selects the trust values for Samsung experts
    # * This applies the trust factor to each expert's weights
    samsung_expert_opinion_weights_trust = samsung_expert_opinion_weights[:, :, i_indices] * trust_matrix[0, i_indices]

    ## * Apply trust matrix to expert opinion weights for iPhone

    # Create an array of expert indices for iPhone
    i_indices = np.arange(iphone_expert_opinion_weights.shape[2], dtype=np.int32)

    # Multiply the expert opinion weights by the corresponding trust values for iPhone
    # * Note 1: iphone_expert_opinion_weights is a 3D array where the third dimension represents different experts
    # * trust_matrix[1, i_indices] selects the trust values for iPhone experts
    # * This applies the trust factor to each expert's weights
    iphone_expert_opinion_weights_trust = iphone_expert_opinion_weights[:, :, i_indices] * trust_matrix[1, i_indices]

    ## * Apply trust matrix to expert opinion weights for Xiaomi

    # Create an array of expert indices for Xiaomi
    i_indices = np.arange(xiaomi_expert_opinion_weights.shape[2], dtype=np.int32)

    # Multiply the expert opinion weights by the corresponding trust values for Xiaomi
    # * Note 1: xiaomi_expert_opinion_weights is a 3D array where the third dimension represents different experts
    # * trust_matrix[2, i_indices] selects the trust values for Xiaomi experts
    # * This applies the trust factor to each expert's weights
    xiaomi_expert_opinion_weights_trust = xiaomi_expert_opinion_weights[:, :, i_indices] * trust_matrix[2, i_indices]

    
    ## Initialize cumulative expert weights
    # * Creating a 3D array to hold the cumulative weights for each brand and feature
    # * Shape: (`number of features`, score_range, `number of brands`)
    # * - The first dimension represents the features.
    # * - The second dimension (score_range) represents the possible scores for each feature.
    # * - The third dimension represents the brands: Samsung, iPhone, and Xiaomi.
    cumulative_expert_weights = np.zeros((3, score_range, 3), dtype=np.float32)

    # Summing the trusted expert opinion weights for Samsung across all experts
    # and storing the result in the first slice of the cumulative weights array.
    # * Note 1: Axis 2 represents summing over all experts for each feature and score.
    cumulative_expert_weights[:, :, 0] = np.sum(samsung_expert_opinion_weights_trust, axis=2)

    # Summing the trusted expert opinion weights for iPhone across all experts
    # and storing the result in the second slice of the cumulative weights array.
    # * Note 1: Axis 2 represents summing over all experts for each feature and score.
    cumulative_expert_weights[:, :, 1] = np.sum(iphone_expert_opinion_weights_trust, axis=2)

    # Summing the trusted expert opinion weights for Xiaomi across all experts
    # and storing the result in the third slice of the cumulative weights array.
    # * Note 1: Axis 2 represents summing over all experts for each feature and score.
    cumulative_expert_weights[:, :, 2] = np.sum(xiaomi_expert_opinion_weights_trust, axis=2)


    # Update primary modeler's weights with cumulative expert weights
    primary_modeler_updated_weights = cumulative_expert_weights + primary_modeler_weights

    # Compute initial posterior probabilities for the primary modeler
    _, primary_modeler_posterior_initial = compute_primary_modeler_posterior_brands(primary_modeler_weights, primary_modeler_brand_pref, primary_modeler_score_preference)
    
    # Compute updated posterior probabilities for the primary modeler
    _, primary_modeler_posterior_updated = compute_primary_modeler_posterior_brands(primary_modeler_updated_weights, primary_modeler_brand_pref, primary_modeler_score_preference)

    return primary_modeler_posterior_initial, primary_modeler_posterior_updated

