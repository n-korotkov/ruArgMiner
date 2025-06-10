from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy

class Text:
    pass
class Project:
    pass
class Statement:
    pass
class Argument:
    pass
class Comment:
    pass
class Corpus:
    pass

@dataclass
class Text:
    id: str
    title: str
    content: str
    comments: list[Comment]
    projects: list[Project]

    def __repr__(self):
        return f'Text[{self.id}]'

    def __hash__(self):
        return hash(self.id)

    def from_dict(text):
        content = ''.join([c if ord(c) < 0x10000 else c * 2 for c in text['text']['content']])
        ret = Text(text['text']['id'], text['text']['Название'], content, [], [])
        ret.comments.extend([Comment.from_dict(comment, n, ret) for n, comment in enumerate(text['text'].get('comments') or [])])
        for project in text['projects']:
            try:
                ret.projects.append(Project.from_dict(project, ret))
            except Exception as e:
                print(e)
                print(f'Excluding project "{project["title"]}" from text "{text["text"]["Название"]}"')
        return ret

@dataclass
class Project:
    id: str
    title: str
    statements: list[Statement]
    arguments: list[Argument]
    equivalents: defaultdict[set]
    text: Text

    def __repr__(self):
        return f'Project[{self.id}]'

    def __hash__(self):
        return hash(self.id)

    def from_dict(project, text):
        ret = Project(project['id'], project['title'], [], [], defaultdict(set), text)
        node_stmts = project['nodes']['statements']
        data_stmts = project['data']['statements']
        node_args = project['nodes']['arguments']
        data_args = project['data']['arguments']
        stmts = {stmt['uri']: Statement.from_dict(data_stmts[stmt['uri']], stmt, ret) for stmt in node_stmts}
        args = {arg['uri']: Argument.from_dict(data_args[arg['uri']], arg, ret) for arg in node_args}
        for uri1, uri2 in project['data']['equivalents']:
            ret.equivalents[stmts[uri1]].add(stmts[uri2])
            # ret.equivalents[stmts[uri2]].add(stmts[uri1])
        for arg in args.values():
            arg.conclusion = stmts.get(arg.conclusion) if arg.is_statement else args.get(arg.conclusion)
            arg.premises = [stmts[uri] for uri in arg.premises]
        ret.statements.extend(stmts.values())
        ret.arguments.extend(args.values())
        return ret

@dataclass
class Statement:
    id: str
    val: str
    range: list[(int, int)]
    record: Text|Comment
    project: Project

    def __repr__(self):
        return f'Statement[{self.id}]'

    def __hash__(self):
        return hash(self.id)

    def from_dict(stmt_data, stmt_node, project):
        range_ = [tuple(span) for span in stmt_node.get('range') or []]
        com = stmt_node.get('com')
        record = project.text.comments[com] if com is not None else project.text
        return Statement(stmt_node['_id'], stmt_data['val'], range_, record, project)

@dataclass
class Argument:
    id: str
    val: str
    is_support: bool
    is_statement: bool
    conclusion: Statement|Argument
    premises: list[Statement]
    project: Project

    def __repr__(self):
        return f'Argument[{self.id}]'

    def __hash__(self):
        return hash(self.id)

    def from_dict(arg_data, arg_node, project):
        is_support = arg_data['type'].endswith('Inference')
        is_statement = False
        conclusion = None
        conclusion_key = 'hasConclusion' if is_support else 'hasConflictedElement'
        if conclusion_key in arg_data['params']:
            is_statement = arg_data['params'][conclusion_key]['is_statement']
            conclusion = arg_data['params'][conclusion_key]['param']
        premises = [d['param'] for key, d in arg_data['params'].items() if key != conclusion_key]
        return Argument(arg_node['_id'], arg_data['val'], is_support, is_statement, conclusion, premises, project)

@dataclass
class Comment:
    n: int
    content: str
    indent: int
    text: Text

    def __repr__(self):
        return f'Comment[{self.text.id}#{self.n}]'

    def __hash__(self):
        return hash((self.text.id, self.n))

    def from_dict(comment, n, text):
        return Comment(n, comment['content'], comment['indent'], text)

@dataclass
class Corpus:
    name: str
    id: str
    texts: list[Text]
    subcorpora: list[Corpus]

    def from_dict(corpus):
        return Corpus(
            corpus['corpus']['Название'],
            corpus['corpus']['id'],
            [Text.from_dict(text) for text in corpus['texts']],
            [Corpus.from_dict(subcorpus) for subcorpus in corpus['subcorpora']],
        )

    def flatten(corpus):
        flat_corpus = Corpus(corpus.name, corpus.id, deepcopy(corpus.texts), [])
        for subcorpus in corpus.subcorpora:
            flat_corpus.texts.extend(Corpus.flatten(subcorpus).texts)
        return flat_corpus

    def merge(corpus0, corpus1):
        return Corpus(
            f'{corpus0.name} + {corpus1.name}',
            f'{corpus0.id} + {corpus1.id}',
            deepcopy(corpus0.texts) + deepcopy(corpus1.texts),
            deepcopy(corpus0.subcorpora) + deepcopy(corpus1.subcorpora),
        )

# @dataclass
# class CorpusDB:
#     corpus: Corpus
#     text_by_id: dict[str, Text]
#     project_by_id: dict[str, Project]
#     statement_by_id: dict[str, Statement]
#     argument_by_id: dict[str, Argument]

#     def from_corpus(corpus: Corpus):
#         db = CorpusDB(corpus, {}, {}, {}, {})
#         for text in corpus.texts:
#             db.text_by_id[text.id] = text
#             for project in text.projects:
#                 db.project_by_id[project.id] = project
#                 for statement in project.statements:
#                     db.statement_by_id[statement.id] = statement
#                 for argument in project.arguments:
#                     db.argument_by_id[argument.id] = argument
#         return db
