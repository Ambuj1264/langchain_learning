from typing import TypedDict

class Movie(TypedDict):
    name: str
    year: int
    rating: float


myMovies = Movie(name="Inception", year=2010, rating=8.8)
print(myMovies)