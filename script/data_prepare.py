from tools.load_data import get_list, process
from tools.utils import get_root_path
import os

root_path = get_root_path()
cat = ['all', 'film-animation', 'autos-vehicles', 'music', 'pets-animals', 'sports', 'travel-events', 'gaming',
'people-blogs', 'comedy', 'entertainment', 'news-politics', 'howto-style', 'education', 'science-technology', 'shows', 'nonprofits-activism']
cat = ['all']
for cls in cat:
    cat_url = f'https://us.youtubers.me/global/{cls}/top-1000-most-subscribed-youtube-channels'
    cat_df = get_list(cat_url)
    cat_df = process(cat_df)
    avg = cat_df['freq_m'].mean()
    globals()[f'{cls}_df'] = cat_df
    globals()[f'{cls}_df'].to_csv(os.path.join(root_path, f'data/{cls}1000.csv'), index=False)