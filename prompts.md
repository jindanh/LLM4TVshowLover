# TV Show Recommendation Prompts

## Basic Recommendation
```prompt
Based on the following TV show preferences: {preferences}
Please recommend a TV show and explain why it would be a good match.
```

## Detailed Recommendation
```prompt
I'm looking for a TV show with the following preferences:
- Genre: {genre}
- Mood: {mood}
- Length: {length}
- Additional preferences: {additional_preferences}

Please recommend a TV show that matches these criteria and explain why it would be a good fit.
Also, provide a brief summary of the show without spoilers.
```

## Comparison Recommendation
```prompt
I really enjoyed watching {favorite_show}. 
Can you recommend similar TV shows that have:
- Similar themes or style: {similar_elements}
- Different aspects I'm looking for: {different_elements}

Please explain why each recommendation would be a good match based on my preferences.
```

## Mood Based Recommendation
```prompt
I'm in the mood for a TV show that makes me feel {mood}.
I'm particularly interested in shows that are {specific_characteristics}.

Please recommend a TV show that matches this mood and explain why it would be perfect for my current state.
```

# Test Queries

## Basic Recommendation
```json
[
    {"preferences": "I like action-packed shows with strong female leads"},
    {"preferences": "I enjoy psychological thrillers with plot twists"},
    {"preferences": "Looking for a light-hearted comedy series"}
]
```

## Detailed Recommendation
```json
[
    {
        "genre": "Science Fiction",
        "mood": "thought-provoking",
        "length": "multiple seasons",
        "additional_preferences": "with strong character development"
    }
]
```

## Comparison Recommendation
```json
[
    {
        "favorite_show": "Breaking Bad",
        "similar_elements": "complex characters and moral dilemmas",
        "different_elements": "more focus on family dynamics"
    }
]
```

## Mood Based Recommendation
```json
[
    {
        "mood": "inspired and motivated",
        "specific_characteristics": "about overcoming challenges and personal growth"
    }
]
``` 