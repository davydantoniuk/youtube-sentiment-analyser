## Models Used

1. **Spam Detection Model** (Model 1): Classifies comments as spam or not spam.
2. **Hate Speech Classifier** (Model 2): Predicts whether a comment is hate speech, offensive, or neither.
3. **Sarcasm Detection Model** (Model 3): Determines if a comment is sarcastic.
4. **Sentiment Analysis Model** (Model 4): Classifies comments as positive, neutral, or negative.

## Decision Process

The final classification is computed using a weighted approach based on the predictions from all four models.

### Step 1: Spam Filtering

- If the probability of spam (**P\_spam**) is greater than 0.8, the comment is discarded.

### Step 2: Hate Speech Adjustment

- If **P\_hate** > 0.7, the sentiment is automatically classified as **negative**.
- If **P\_offensive** > 0.7, the weight of **positive** sentiment is reduced.

### Step 3: Sarcasm Adjustment

- If **P\_sarcasm** > 0.7, the sentiment probabilities from Model 4 are adjusted:
  - **Positive sentiment** confidence is reduced.
  - **Neutral sentiment** remains unchanged.
  - **Negative sentiment** confidence is increased.

### Step 4: Weighted Sentiment Calculation

The final sentiment score is computed as:

$$
S = w_1 P_{positive} + w_2 P_{neutral} + w_3 P_{negative}
$$

where the default weights are:

- **w\_1 = 1, w\_2 = 1, w\_3 = 1** (no adjustments)
- If sarcasm is detected, adjust:
  - **w\_1 = 0.5, w\_2 = 1.2, w\_3 = 1.1**
- If hate speech is detected, increase **w\_3** and decrease **w\_1**.

### Step 5: Final Classification

The sentiment with the highest weighted probability determines the classification.

## Example Calculation

Given a comment with the following probabilities:

- **Spam Detection:** P\_spam = 0.3 (not spam)
- **Hate Speech:** P\_hate = 0.2, P\_offensive = 0.8, P\_neither = 0.0
- **Sarcasm Detection:** P\_sarcasm = 0.9
- **Sentiment Analysis:** P\_positive = 0.6, P\_neutral = 0.3, P\_negative = 0.1

Since sarcasm is high (**P\_sarcasm > 0.7**), we adjust weights:

- **w\_1 = 0.5, w\_2 = 1.2, w\_3 = 1.1**

Final sentiment calculation:

$$
S_{positive} = 0.5 \times 0.6 = 0.3
$$

$$
S_{neutral} = 1.2 \times 0.3 = 0.36
$$

$$
S_{negative} = 1.1 \times 0.1 = 0.11
$$

Since **S\_neutral** is the highest, the final classification is **Neutral**.

