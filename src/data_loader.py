
import glob
import pandas as pd
from typing import List, Optional

def load_and_preprocess(
    path_pattern: str,
    text_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None
) -> pd.DataFrame:

    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"Нет файлов по шаблону {path_pattern}")
    df_list = [pd.read_excel(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    if date_col:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        else:
            print(f"Warning: колонка даты '{date_col}' не найдена — пропускаем.")

    if text_cols is None:
        text_cols = ['advantages', 'disadvantages', 'comment']

    def combine_text(row):
        parts = []
        for col in text_cols:
            val = row.get(col, "")
            if pd.notna(val):
                s = str(val).strip()
                if s:
                    parts.append(s)
        return ". ".join(parts)

    df['text'] = df.apply(combine_text, axis=1)

    return df
