from dataclasses import dataclass
from collections import defaultdict
from re import finditer

from corpus_utils import *

@dataclass
class Paragraph:
    span: tuple[int, int]
    fragments: list[(int, int)]

@dataclass
class TextSegmentation:
    text: Text
    paragraphs_by_record: defaultdict[Text|Comment, list[Paragraph]]

def segment_texts(texts: list[Text], segment_fn) -> list[TextSegmentation]:
    def in_brackets(s, start, end, left_bracket, right_bracket):
        return s.rfind(right_bracket, start, end) < s.rfind(left_bracket, start, end)

    def in_any_brackets(s, start, end):
        for left_bracket, right_bracket in ['()', '[]', '{}']:
            if in_brackets(s, start, end, left_bracket, right_bracket):
                return True
        return False

    segmentations = []
    paragraph_contents = []
    contexts = []
    for text in texts:
        segmentations.append(TextSegmentation(text, defaultdict(list)))
        for record in [text] + text.comments:
            for match in finditer(r'\s*[^\n]+\s*', record.content):
                b, e = match.span()
                paragraph = Paragraph((b, e), [])
                segmentations[-1].paragraphs_by_record[record].append(paragraph)
                paragraph_contents.append(record.content[b:e])
                contexts.append((paragraph, record))

    for (sents, (para, record)) in zip(segment_fn(paragraph_contents), contexts):
        for b, e in sents:
            b += para.span[0]
            e += para.span[0]
            if para.fragments != [] and in_any_brackets(record.content, *para.fragments[-1]):
                para.fragments[-1] = (para.fragments[-1][0], e)
            else:
                para.fragments.append((b, e))

    return segmentations
