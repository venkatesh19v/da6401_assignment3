import unicodedata
from collections import Counter, defaultdict
import csv

TAMIL_VOWELS = set("அஆஇஈஉஊஎஏஐஒஓஔஃஂஃ்")
TAMIL_CONSONANTS = set("கஙசஞடணதநபமயரலவழளறனஜஷஸஹ")

def is_tamil_vowel(ch):
    return ch in TAMIL_VOWELS or 'VOWEL' in unicodedata.name(ch, '')

def is_tamil_consonant(ch):
    return ch in TAMIL_CONSONANTS or 'CONSONANT' in unicodedata.name(ch, '')

def analyze_tamil_transliteration(tsv_path):
    vowel_errors = 0
    consonant_errors = 0
    confused_pairs = Counter()
    total_errors = 0
    total_words = 0
    correct_words = 0

    with open(tsv_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) 
        for row in reader:
            if len(row) < 3:
                continue
            _, ref, pred = row
            total_words += 1
            if ref == pred:
                correct_words += 1
                continue
            for r_ch, p_ch in zip(ref, pred):
                if r_ch != p_ch:
                    total_errors += 1
                    confused_pairs[(p_ch, r_ch)] += 1
                    if is_tamil_vowel(r_ch) and is_tamil_vowel(p_ch):
                        vowel_errors += 1
                    elif is_tamil_consonant(r_ch) and is_tamil_consonant(p_ch):
                        consonant_errors += 1

    print(f"\nTotal words: {total_words}")
    print(f"Correctly predicted words: {correct_words}")
    print(f"Word-level accuracy: {correct_words / total_words:.2%}")
    print(f"Total character mismatches: {total_errors}")
    print(f"Vowel errors: {vowel_errors}")
    print(f"Consonant errors: {consonant_errors}")
    print("\nTop 10 confused character pairs (predicted → actual):")
    for (pred, actual), count in confused_pairs.most_common(10):
        print(f"  '{pred}' → '{actual}': {count} times")

analyze_tamil_transliteration("van_gru_test_predictions.tsv")
