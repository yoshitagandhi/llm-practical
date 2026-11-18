# Task 3 – outputs side by side

Score each output 0 (wrong / ignores instructions), 1 (partly right), 2 (correct and follows instructions).


## Prompt 1 (factual)

**Prompt:** What is the capital of Australia? Answer in one word.

**Expected:** Canberra


**fp16** (3 tokens, 5.59 tok/s) – score: __

```
Canberra
```


**q4_K_M** (3 tokens, 40.68 tok/s) – score: __

```
Canberra
```


## Prompt 2 (factual)

**Prompt:** Who wrote the novel Pride and Prejudice, and in which century was it published?

**Expected:** Jane Austen, 19th century (1813)


**fp16** (23 tokens, 9.76 tok/s) – score: __

```
Jane Austen wrote the novel "Pride and Prejudice" in the 19th century.
```


**q4_K_M** (71 tokens, 24.29 tok/s) – score: __

```
The novel "Pride and Prejudice" was written by Jane Austen, a renowned English novelist. It was published in 1813. Jane Austen was a significant figure in the English literature of the 19th century, known for her witty and insightful portrayals of the social norms and class distinctions of her time.
```


## Prompt 3 (factual)

**Prompt:** In two sentences, explain what photosynthesis is.

**Expected:** Plants use light, water and CO2 to make glucose and release oxygen; exactly two sentences


**fp16** (22 tokens, 7.75 tok/s) – score: __

```
Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen.
```


**q4_K_M** (22 tokens, 27.85 tok/s) – score: __

```
Photosynthesis is the process by which plants convert light energy from the sun into chemical energy stored in glucose.
```


## Prompt 4 (reasoning)

**Prompt:** A shop sells pens at 3 for 20 rupees. How much do 12 pens cost? Show your working briefly.

**Expected:** 80 rupees (12 / 3 = 4 groups, 4 x 20 = 80)


**fp16** (256 tokens, 9.15 tok/s) – score: __

```
To find out how much 12 pens cost, we can follow these steps:

1. **Determine the cost of one pen:**
   - The shop sells 3 pens for 20 rupees.
   - Therefore, the cost of one pen is \( \frac{20 \text{ rupees}}{3 \text{ pens}} \).

2. **Calculate the cost of 12 pens:**
   - Since one pen costs \( \frac{20}{3} \) rupees, the cost of 12 pens would be \( 12 \times \frac{20}{3} \) rupees.

Let's do the calculation:

\[ \text{Cost of one pen} = \frac{20}{3} \text{ rupees} \]
\[ \text{Cost of 12 pens} = 12 \times \frac{20}{3} \text{ rupees} \]
\[ \text{Cost of 12 pens} = 12 \times \frac{20}{3} = 4 \times 20 = 80 \text{ rupees} \]

So, 12 pens cost 80
```


**q4_K_M** (160 tokens, 25.49 tok/s) – score: __

```
To find out how much 12 pens cost, we first need to determine the cost of one pen. Since 3 pens cost 20 rupees, we can calculate the cost of one pen by dividing 20 rupees by 3:

Cost of one pen = 20 rupees / 3 = 6.67 rupees (rounded to two decimal places)

Now that we know the cost of one pen, we can calculate the cost of 12 pens by multiplying the cost of one pen by 12:

Cost of 12 pens = 6.67 rupees/pen * 12 pens = 80.04 rupees

So, 12 pens cost 80.04 rupees.
```


## Prompt 5 (reasoning)

**Prompt:** If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes or no, then explain in one sentence.

**Expected:** Yes, with a correct one-sentence explanation


**fp16** (47 tokens, 9.23 tok/s) – score: __

```
Yes, all bloops are definitely lazzies because if all bloops are razzies and all razzies are lazzies, then the set of bloops is a subset of the set of lazzies.
```


**q4_K_M** (67 tokens, 24.72 tok/s) – score: __

```
Yes, all bloops are definitely lazzies. This is because if all bloops are razzies and all razzies are lazzies, then every bloop must be a razzie, and since all razzies are lazzies, it follows that every bloop is a lazzie.
```


## Prompt 6 (code)

**Prompt:** Write a Python function is_palindrome(s) that returns True if s is a palindrome, ignoring case and spaces. Return only the code.

**Expected:** Correct function, e.g. normalise with lower() and remove spaces, compare with reverse; code only


**fp16** (111 tokens, 9.58 tok/s) – score: __

```
```python
def is_palindrome(s):
    """
    Check if the given string is a palindrome, ignoring case and spaces.
    
    Args:
    s (str): The string to check.
    
    Returns:
    bool: True if s is a palindrome, False otherwise.
    """
    # Remove spaces and convert to lowercase
    cleaned_s = ''.join(c.lower() for c in s if c.isalnum())
    # Check if the cleaned string is equal to its reverse
    return cleaned_s == cleaned_s[::-1]
```
```


**q4_K_M** (109 tokens, 21.26 tok/s) – score: __

```
```python
def is_palindrome(s):
    """
    Check if the given string is a palindrome, ignoring case and spaces.
    
    Parameters:
    s (str): The string to check.
    
    Returns:
    bool: True if the string is a palindrome, False otherwise.
    """
    # Remove spaces and convert to lowercase
    cleaned_s = ''.join(filter(str.isalnum, s)).lower()
    # Check if the cleaned string is equal to its reverse
    return cleaned_s == cleaned_s[::-1]
```
```


## Prompt 7 (code)

**Prompt:** What does this Python code print?

x = [1, 2, 3]
y = x
y.append(4)
print(len(x))

**Expected:** 4 (y and x refer to the same list)


**fp16** (256 tokens, 9.16 tok/s) – score: __

```
The Python code you provided will print `2`. Here's a step-by-step explanation of what happens:

1. **Initialization of Lists**:
   ```python
   x = [1, 2, 3]
   y = x
   ```
   - `x` is initialized with the list `[1, 2, 3]`.
   - `y` is initialized with the same list `[1, 2, 3]`.

2. **Appending to List `y`**:
   ```python
   y.append(4)
   ```
   - The `append` method is called on list `y`, which modifies `y` to include the element `4` at the end of the list.
   - After this operation, `y` becomes `[1, 2, 3, 4]`.

3. **Printing the Length of List `x`**:
   ```python
   print(len(x))
   ```
   - The `len` function is called on list `x`, which returns the number of elements in the list.
   - Since `x` still contains the elements `[1, 2, 3]`, the length of `x` is `3`.

Therefore, the output of the
```


**q4_K_M** (221 tokens, 23.5 tok/s) – score: __

```
The Python code you provided will print `2`. Here's a step-by-step explanation of what happens:

1. `x = [1, 2, 3]`: This line creates a list `x` containing the elements `1`, `2`, and `3`.

2. `y = x`: This line creates a new variable `y` and assigns it the same list `x` that was created in the first line. So `y` is also `[1, 2, 3]`.

3. `y.append(4)`: This line appends the integer `4` to the list `y`. After this line, `y` will be `[1, 2, 3, 4]`.

4. `print(len(x))`: This line prints the length of the list `x`. Since `x` is still `[1, 2, 3]`, its length is `3`.

Therefore, the output of the code will be `2`, which is the length of the list `x` after the append operation.
```


## Prompt 8 (format)

**Prompt:** List the three primary colours of light as a JSON array of strings. Output only the JSON.

**Expected:** ["red", "green", "blue"] with nothing else


**fp16** (14 tokens, 7.57 tok/s) – score: __

```
```json
["red", "green", "blue"]
```
```


**q4_K_M** (14 tokens, 19.83 tok/s) – score: __

```
```json
["red", "green", "blue"]
```
```


## Prompt 9 (format)

**Prompt:** Write exactly three bullet points on why sleep matters. Each bullet must be under 10 words.

**Expected:** Exactly 3 bullets, each under 10 words


**fp16** (27 tokens, 9.46 tok/s) – score: __

```
- Sleep repairs and rejuvenates the body.
- It helps in memory consolidation and learning.
- It regulates mood and emotional balance.
```


**q4_K_M** (25 tokens, 28.65 tok/s) – score: __

```
- Sleep repairs and rejuvenates the body.
- It enhances memory and learning.
- It improves mood and emotional balance.
```


## Prompt 10 (summary)

**Prompt:** Summarise the following in one sentence:

The city council voted on Tuesday to extend the library's opening hours to 9 pm on weekdays, starting next month. The change follows a survey in which most students said they wanted somewhere quiet to study in the evening. The extra hours will be paid for by moving money from the events budget.

**Expected:** One sentence covering: council extends library hours to 9 pm on weekdays, because students asked for evening study space, funded from events budget


**fp16** (30 tokens, 8.57 tok/s) – score: __

```
The city council extended the library's opening hours to 9 pm on weekdays, with the extra cost coming from a reduction in the events budget.
```


**q4_K_M** (27 tokens, 27.14 tok/s) – score: __

```
The city council voted to extend the library's weekday hours to 9 pm, with the extra cost coming from a budget shift.
```
