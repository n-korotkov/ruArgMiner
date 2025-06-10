from dataclasses import dataclass

from corpus_utils import *
from segmentation import *

@dataclass
class ParagraphLoc:
    span: tuple[int, int]
    record: Text|Comment
    fragment_span: tuple[int, int]
    def content(self):
        return self.record.content[slice(*self.span)]

@dataclass
class FragmentLoc:
    span: tuple[int, int]
    record: Text|Comment
    paragraph_index: int
    covered_statements: list[Statement]
    def content(self):
        return self.record.content[slice(*self.span)]

@dataclass
class StatementLoc:
    span: tuple[int, int]
    statement: Statement

@dataclass
class SimpleArgStructure:
    project: Project
    paragraphs: list[ParagraphLoc]
    fragments: list[FragmentLoc]
    supports: defaultdict[int, set[int]]
    attacks: defaultdict[int, set[int]]
    related_paragraphs: defaultdict[int, set[int]]
    related_fragments: defaultdict[int, set[int]]

def build_simple_structure(project: Project, segmentation: TextSegmentation, stmt_frag_intersect_fn):
    ordered_stmt_locs = defaultdict(list[StatementLoc])
    for stmt in project.statements:
        for span in stmt.range:
            ordered_stmt_locs[stmt.record].append(StatementLoc(span, stmt))

    for stmts in ordered_stmt_locs.values():
        stmts.sort(key=lambda x: x.span)
        stmts.append(StatementLoc((1e18, 1e18), None))

    fragments_by_stmt = defaultdict(list)
    paragraphs = []
    fragments = []
    for record, stmt_locs in ordered_stmt_locs.items():
        first_stmt_index = 0
        for para in segmentation.paragraphs_by_record[record]:
            for (start, end) in para.fragments:
                covered_statements = []
                if stmt_locs != []:
                    while start >= stmt_locs[first_stmt_index].span[1]:
                        first_stmt_index += 1
                    stmt_index = first_stmt_index
                    while stmt_locs[stmt_index].span[0] < end:
                        stmt = stmt_locs[stmt_index].statement
                        if stmt_frag_intersect_fn(record, stmt_locs[stmt_index].span, (start, end)):
                            covered_statements.append(stmt)
                            fragments_by_stmt[stmt].append(len(fragments))
                        stmt_index += 1
                fragments.append(FragmentLoc((start, end), record, len(paragraphs), covered_statements))
            first_frag_index = paragraphs[-1].fragment_span[1] if len(paragraphs) > 0 else 0
            paragraphs.append(ParagraphLoc(para.span, record, (first_frag_index, first_frag_index + len(para.fragments))))

    supports = defaultdict(set)
    attacks = defaultdict(set)
    related_paragraphs = defaultdict(set)
    related_fragments = defaultdict(set)

    for arg in project.arguments:
        premises = arg.premises
        is_support = arg.is_support
        while not arg.is_statement and arg.conclusion is not None:
            arg = arg.conclusion
            is_support = (is_support == arg.is_support)
        if arg.conclusion is not None:
            if arg.conclusion in premises:
                print(arg)
            for premise in premises:
                for i in fragments_by_stmt[premise]:
                    for j in fragments_by_stmt[arg.conclusion]:
                        related_fragments[i].add(j)
                        related_fragments[j].add(i)
                        related_paragraphs[fragments[i].paragraph_index].add(fragments[j].paragraph_index)
                        related_paragraphs[fragments[j].paragraph_index].add(fragments[i].paragraph_index)
                        (supports if is_support else attacks)[i].add(j)

    return SimpleArgStructure(project, paragraphs, fragments, supports, attacks, related_paragraphs, related_fragments)
