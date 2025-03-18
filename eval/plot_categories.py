
# [({66, 37, 40, 46, 58, 27}, {17, 18, 19, 20, 21, 22})] - ['feed', 'hold', 'hug', 'kiss', 'no_interaction', 'pet'], ['cat', 'dog', 'horse', 'sheep', 'cow', 'elephant']
# [({37, 9, 16, 24, 58}, {57, 58, 59, 61, 52})] - ['cut', 'hold', 'eat', 'carry', 'no_interaction'], ['carrot', 'hot dog', 'pizza', 'cake', 'banana']
# [({37, 9, 55, 24, 58}, {58, 59, 60, 61, 54})] - ['hold', 'make', 'eat', 'carry', 'no_interaction'], ['hot dog', 'pizza', 'donut', 'cake', 'sandwich']
# [({37, 9, 44, 77, 58}, {2, 35, 36, 41, 42})] - ['hold', 'carry', 'no_interaction', 'jump', 'ride'], ['skateboard', 'surfboard', 'bicycle', 'skis', 'snowboard']
triplets_text_grid = {
    (1, 27, 17): 'a photo of a person feeding a cat', (1, 37, 17): 'a photo of a person holding a cat', (1, 72, 17): 'a photo of a person pushing a cat', (1, 111, 17): 'a photo of a person taking a cat out for a walk', (1, 66, 17): 'a photo of a person petting a cat', (1, 69, 17): 'a photo of a person pointing a cat',
    (1, 27, 21): 'a photo of a person feeding a cow', (1, 37, 21): 'a photo of a person holding a cow', (1, 72, 21): 'a photo of a person pushing a cow', (1, 111, 21): 'a photo of a person taking a cow out for a walk', (1, 66, 21): 'a photo of a person petting a cow', (1, 69, 21): 'a photo of a person pointing a cow',
    (1, 37, 18): 'a photo of a person holding a dog', (1, 27, 18): 'a photo of a person feeding a dog', (1, 72, 18): 'a photo of a person pushing a dog', (1, 111, 18): 'a photo of a person taking a dog out for a walk', (1, 66, 18): 'a photo of a person petting a dog', (1, 69, 18): 'a photo of a person pointing with the hand a dog', (1, 27, 19): 'a photo of a person feeding a horse',
    (1, 37, 19): 'a photo of a person holding a horse', (1, 72, 19): 'a photo of a person pushing a horse', (1, 111, 19): 'a photo of a person taking a horse out for a walk', (1, 66, 19): 'a photo of a person petting a horse', (1, 69, 19): 'a photo of a person pointing a horse',
    (1, 27, 20): 'a photo of a person feeding a sheep', (1, 37, 20): 'a photo of a person holding a sheep', (1, 72, 20): 'a photo of a person pushing a sheep', (1, 111, 20): 'a photo of a person taking a sheep out for a walk', (1, 66, 20): 'a photo of a person petting a sheep', (1, 69, 20): 'a photo of a person pointing with the hand a sheep',
    (1, 27, 22): 'a photo of a person feeding an elephant', (1, 37, 22): 'a photo of a person holding an elephant', (1, 72, 22): 'a photo of a person pushing an elephant', (1, 111, 22): 'a photo of a person taking a elephant out for a walk', (1, 66, 22): 'a photo of a person petting an elephant', (1, 69, 22): 'a photo of a person pointing an elephant',

    (1, 16, 53): 'a photo of a person cutting an apple with a knife', (1, 24, 53): 'a photo of a person eating an apple', (1, 37, 53): 'a photo of a person holding an apple', (1, 105, 53): 'a photo of a person throwing an apple',
    (1, 16, 84): 'a photo of a person cutting a book with a knife', (1, 24, 84): 'a photo of a person eating a book', (1, 37, 84): 'a photo of a person holding a book', (1, 105, 84): 'a photo of a person throwing a book', (1, 55, 84): 'a photo of a person making a book',
    (1, 16, 61): 'a photo of a person cutting a cake with a knife', (1, 24, 61): 'a photo of a person eating a cake', (1, 37, 61): 'a photo of a person holding a cake', (1, 55, 61): 'a photo of a person making a cake', (1, 105, 61): 'a photo of a person throwing cake',
    (1, 16, 60): 'a photo of a person cutting a donut with a knife', (1, 24, 60): 'a photo of a person eating a donut', (1, 37, 60): 'a photo of a person holding a donut', (1, 55, 60): 'a photo of a person making a donut', (1, 105, 60): 'a photo of a person throwing a donut',
    (1, 16, 58): 'a photo of a person cutting a hot dog with a knife', (1, 24, 58): 'a photo of a person eating a hot dog', (1, 37, 58): 'a photo of a person holding a hot dog', (1, 55, 58): 'a photo of a person making a hot dog', (1, 105, 58): 'a photo of a person throwing a hot dog',
    (1, 16, 59): 'a photo of a person cutting a pizza with a knife', (1, 24, 59): 'a photo of a person eating a pizza', (1, 37, 59): 'a photo of a person holding a pizza', (1, 55, 59): 'a photo of a person making a pizza', (1, 105, 59): 'a photo of a person throwing a pizza',
    (1, 16, 54): 'a photo of a person cutting a sandwich with a knife', (1, 24, 54): 'a photo of a person eating a sandwich', (1, 37, 54): 'a photo of a person holding a sandwich', (1, 55, 54): 'a photo of a person making a sandwich', (1, 105, 54): 'a photo of a person throwing a sandwich',

    (1, 9, 2): 'a photo of a person carrying a bicycle', (1, 37, 2): 'a photo of a person holding in hand a bicycle', (1, 44, 2): 'a photo of a person jumping with a bicycle', (1, 77, 2): 'a photo of a person riding a bicycle', (1, 69, 2): 'a photo of a person pointing with the hand a bicycle',
    (1, 9, 41): 'a photo of a person carrying a skateboard', (1, 37, 41): 'a photo of a person holding in hand a skateboard', (1, 44, 41): 'a photo of a person jumping with a skateboard', (1, 77, 41): 'a photo of a person riding a skateboard',(1, 69, 41): 'a photo of a person pointing with the hand a skateboard',
    (1, 9, 35): 'a photo of a person carrying snow skis blades in hand', (1, 37, 35): 'a photo of a person holding in hand a snow skis blade', (1, 44, 35): 'a photo of a person jumping with snow skis blades', (1, 77, 35): 'a photo of a person riding snow skis blades', (1, 69, 35): 'a photo of a person pointing with the hand a snow skis blade',
    (1, 9, 36): 'a photo of a person carrying a snowboard', (1, 37, 36): 'a photo of a person holding in hand a snowboard', (1, 44, 36): 'a photo of a person jumping with a snowboard', (1, 77, 36): 'a photo of a person riding a snowboard', (1, 69, 36): 'a photo of a person pointing with the hand a snowboard',
    (1, 9, 42): 'a photo of a person carrying a surfboard', (1, 37, 42): 'a photo of a person holding in hand a surfboard', (1, 44, 42): 'a photo of a person jumping with a surfboard', (1, 77, 42): 'a photo of a person riding a surfboard', (1, 69, 42): 'a photo of a person pointing with the hand a surfboard'
}
llava_query_grid = (
    "The image is masked and I want to reimagine the black hidden part to contain '{triplet_text}'. "
    "Provide an OBJECTIVE and DESCRIPTIVE caption for the complete NEW scene including an identifiable description of the action performed in the image by the person, start with `A photo of` and make sure to mention "
    "the 'person' subject of the action to fit the scene, the NEW object of the action `{object}`, and the action `{verb}`. "
    "Make a sentence with present participle verbs and indeterminate article. Max 250 characters allowed"
)

triplets_text_teaser = {
    (1, 24, 54): 'a photo of a person eating a sandwich', (1, 37, 54): 'a photo of a person holding an hamburger', (1, 27, 44): 'a photo of a person drinking water from a bottle',
    (1, 66, 18): 'a photo of a person petting a dog',(1, 108, 18): 'a photo of a person jogging with a dog in the park', 
    (1, 111, 53): 'a photo of a person walking with an apple in hand', (1, 74, 84): 'a photo of a person reading a book',
    (1, 105, 34): 'a photo of a person throwing a frisbee', (1, 105, 37): 'a photo of a person throwing a basketball', (1, 20, 37): 'a photo of a person dribbling with a basketball',
    (1, 77, 2): 'a photo of a person riding with a bicycle', (1, 44, 41): 'a photo of a person jumping with a skateboard', (1, 33, 41): 'a photo of a person performing a trick with a skateboard',
}
llava_query_teaser = (
    "I want to reimagine the image as if it contains '{triplet_text}'. "
    "Provide an OBJECTIVE and DESCRIPTIVE caption for the complete scene including an identifiable description of the action performed in the image by the person, start with `A photo of` and make sure to mention "
    "the 'person' subject of the action to fit the scene, the object of the action `{object}`, and the action `{verb}`. "
    "Make a sentence with present participle verbs and indeterminate article. Max 250 characters allowed"
)
