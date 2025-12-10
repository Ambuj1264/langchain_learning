from pydantic import BaseModel, Field

class Movie(BaseModel):
    name: str
    year: int
    rating: float
    cgpa: float = Field(gt=4.0, lt=10.0)   # type + Field(default/validators)

movie_data = {"name":"55.7", "year":456, "rating":8.8, "cgpa":5.5}

myMovies = Movie(**movie_data)
# print(myMovies)

# myMovies= dict(myMovies)
myMovies=myMovies.model_dump_json()
print(myMovies)
