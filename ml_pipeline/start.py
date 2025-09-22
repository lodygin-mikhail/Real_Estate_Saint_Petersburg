import src

RAW_DATA_PATH = "data/raw/Spb_flats_prices.csv"
CLEANED_DATA_PATH = "data/interim/cleaned_data.csv"
FEATURED_DATA_PATH = "data/interim/added_features.csv"

if __name__ == "__main__":
    src.clean_data(RAW_DATA_PATH, CLEANED_DATA_PATH)
    src.add_features(CLEANED_DATA_PATH, FEATURED_DATA_PATH)
