import json
import os
from dataclasses import dataclass, asdict
import sys

from plankapy.v2 import Planka
import ollama

from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


planka = Planka(os.environ.get("PLANKA_BASE_URL"))
planka.login(
    username=os.environ.get("PLANKA_USER"), password=os.environ.get("PLANKA_PASSWORD")
)

print([a.name for a in planka.projects])
project = planka.projects[
    int(input("Project Index (0-" + str(len(planka.projects) - 1) + "): "))
]

print([a.name for a in project.boards])
board = project.boards[
    int(input("Board Index (0-" + str(len(project.boards) - 1) + "): "))
]

print([a.name for a in board.lists])
lst = board.lists[int(input("List Index (0-" + str(len(board.lists) - 1) + "): "))]

cards = lst.cards


@dataclass
class EmbeddedCard:
    name: str
    card_labels: list
    card_memberships: list
    embeddings: list


def train(cards, data:list[EmbeddedCard]=[]):
    names = [d.name for d in data]
    for i, card in enumerate(cards):
        if card.name not in names:
            data.append(
                EmbeddedCard(
                    card.name,
                    [label.id for label in card.labels],
                    [member.user.id for member in card.card_memberships],
                    ollama.embeddings(model="embeddinggemma", prompt=card.name).embedding,
                )
            )
        print(f"{i + 1}/{len(cards)}")
    return data


def predict(data, cards):
    for i, card in enumerate(cards):
        print(f"{i + 1}/{len(cards)}")
        max_score = -1
        max_card = data[0]
        for embedcard in data:
            score = cosine_similarity(
                embedcard["embeddings"],
                ollama.embeddings("embeddinggemma", card.name).embedding,
            )
            if score > max_score:
                max_score = score
                max_card = embedcard
        print(f"{card.name} - {max_card['name']}: {max_score}")

        for user in board.users:
            if user.id in max_card["card_memberships"]:
                card.add_member(user)

        for label in board.labels:
            if label.id in max_card["card_labels"]:
                card.add_label(label, add_to_board=False)


if sys.argv[1] == "train":
    input_file=input("File Input (optional): ")
    data = []
    if not input_file== "":
        j=json.load(open(input_file))
        for card in j:
            data.append(EmbeddedCard(
                card["name"],
                card["card_labels"],
                card["card_memberships"],
                card["embeddings"]
            ))
    data = train(cards, data)
    json.dump([asdict(d) for d in data], open(input("File Output: "), "w"))

elif sys.argv[1] == "predict":
    data = json.load(open(input("File: ")))
    predict(data, cards)