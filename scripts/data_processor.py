from datasets import load_dataset
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
            5.0: 'positive',
            4.0: 'positive',
            3.0: 'neutral',
            2.0: 'negative',
            1.0: 'negative',
        }
    )
    return df


def balance_ratings(df, class_col='sentiment'):
    g = df.groupby(class_col)
    df = g.apply(lambda x:
                 x.sample(g.size().min()).reset_index(drop=True),
                 include_groups=False)
    return df


def process_data(hf_dataset, split='train'):

    df = hf_dataset[split].to_pandas()

    if df['text'].isnull().sum() > 0:
        df.dropna(subset=['text'], inplace=True)

    # Clean text (remove special chars, standardize whitespace)
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

    # Save to CSV
    df.to_csv(f'../data/data_{split}.csv', index=False)

if __name__ == '__main__':
    dataset_name = 'allegro/klej-allegro-reviews'
    dataset = load_dataset(dataset_name)

    for split in dataset:
        process_data(dataset, split=split)

