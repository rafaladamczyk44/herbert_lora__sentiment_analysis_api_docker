from datasets import load_dataset
import pandas as pd
import re

def clean_text(text):
    """Remove special characters and standardize whitespaces"""
    if not isinstance(text, str):
        return text

    # Remove special characters but keep Polish letters and basic punctuation
    # Keep: letters (including Polish), digits, spaces, and basic punctuation (.,!?-')
    text = re.sub(r'[^\w\s.,!?\-\'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', ' ', text)

    # Standardize whitespaces: replace multiple spaces/tabs/newlines with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text

def clean_dataframe(df, text_col='text'):
    """Apply text cleaning to the text column"""
    df[text_col] = df[text_col].apply(clean_text)
    return df

def change_rating(df, new_class_col):
    """Change 1-5 rating to negative, neutral, positive"""
    df[new_class_col] = df['rating'].map(
        {
            5.0: 0,
            4.0: 0,
            3.0: 1,
            2.0: 2,
            1.0: 2,
        }
    )
    return df


def balance_ratings(df, class_col='sentiment'):
    """Downsample all classes to the size of the minority class"""
    min_size = df[class_col].value_counts().min()

    # Sample min_size from each class
    balanced_dfs = []
    for sentiment_class in df[class_col].unique():
        class_df = df[df[class_col] == sentiment_class].sample(n=min_size, random_state=42)
        balanced_dfs.append(class_df)

    # Concatenate and shuffle
    df_balanced = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    return df_balanced


def process_data(hf_dataset, split='train'):

    df = hf_dataset[split].to_pandas()

    # Drop rows with null values in any column
    df = df.dropna()

    # Drop rows with empty text (empty strings)
    df = df[df['text'].str.strip() != '']

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    # Clean text
    df = clean_dataframe(df, text_col='text')

    # Map ratings to sentiment categories
    df = change_rating(df, 'sentiment')

    # Balance classes (downsample to minority class)

    print(f"Class distribution in {split} before balancing:\n{df['sentiment'].value_counts()}")
    df = balance_ratings(df)
    print(f"Class distribution after balancing:\n{df['sentiment'].value_counts()}")

    # Clean up dataframe
    df = df.reset_index(drop=True)
    df = df.drop(['rating'], axis=1)

    df.rename(columns={'text': 'text', 'sentiment': 'labels'}, inplace=True)

    # Save to CSV
    df.to_csv(f'../data/data_{split}.csv', index=False)

if __name__ == '__main__':
    dataset_name = 'allegro/klej-allegro-reviews'
    dataset = load_dataset(dataset_name)

    for split in dataset:
        process_data(dataset, split=split)

