import json
from ftfy import fix_text
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_helper import llm  # Ensure this points to your LLM setup

def process_posts(raw_file_path='data/raw_posts.json', processed_file_path='data/processed_posts.json'):
    enriched_posts = []

    try:
        with open(raw_file_path, encoding="utf-8") as file:
            posts = json.load(file)
    except UnicodeDecodeError as e:
        print("❌ Unicode decode error:", str(e))
        return
    except json.JSONDecodeError as e:
        print("❌ JSON decode error:", str(e))
        return

    for post in posts:
        if 'text' not in post:
            continue

        post['text'] = fix_text(post['text'])  # fix malformed unicode
        metadata = extract_metadata(post['text'])

        # Merge original post and metadata
        post_with_metadata = {**post, **metadata}
        enriched_posts.append(post_with_metadata)

    unified_tags = get_unified_tags(enriched_posts)

    for post in enriched_posts:
        current_tags = post.get('tags', [])
        post['tags'] = [unified_tags.get(tag, tag) for tag in current_tags]

    try:
        with open(processed_file_path, mode="w", encoding="utf-8") as outfile:
            json.dump(enriched_posts, outfile, indent=4, ensure_ascii=False)
        print(f"✅ Successfully saved {len(enriched_posts)} posts to {processed_file_path}")
    except Exception as e:
        print("❌ Error saving processed posts:", str(e))


def extract_metadata(post):
    MAX_LENGTH = 2000
    if len(post) > MAX_LENGTH:
        post = post[:MAX_LENGTH] + "..."

    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means Hindi + English)
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    json_parser = JsonOutputParser()

    try:
        response = chain.invoke({"post": post})
        return json_parser.parse(response.content)
    except OutputParserException as e:
        print("⚠️ Failed to parse metadata for post:\n", post[:200], "...\nError:", e)
        return {
            "line_count": post.count('\n') + 1,
            "language": "Unknown",
            "tags": []
        }
    except Exception as e:
        print("⚠️ Unexpected error while extracting metadata:", e)
        return {
            "line_count": post.count('\n') + 1,
            "language": "Unknown",
            "tags": []
        }


def get_unified_tags(posts_with_metadata):
    unique_tags = set()
    for post in posts_with_metadata:
        unique_tags.update(post.get('tags', []))

    unique_tags_list = ', '.join(sorted(unique_tags))

    template = '''
    I will give you a list of tags. You need to unify tags with the following requirements:
    1. Tags are unified and merged to create a shorter list.
       Example 1: "Jobseekers", "Job Hunting" → "Job Search"
       Example 2: "Motivation", "Inspiration" → "Motivation"
       Example 3: "Personal Growth", "Self Improvement" → "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" → "Scams"
    2. Use title case. Example: "Job Search", "Motivation"
    3. Output must be a JSON object. No preamble.
    4. Format: {"OldTag1": "UnifiedTag", "OldTag2": "UnifiedTag", ...}

    Here is the list of tags:
    {tags}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    json_parser = JsonOutputParser()

    try:
        response = chain.invoke({"tags": unique_tags_list})
        return json_parser.parse(response.content)
    except OutputParserException as e:
        print("⚠️ Failed to unify tags:", e)
        return {tag: tag for tag in unique_tags}
    except Exception as e:
        print("⚠️ Unexpected error while unifying tags:", e)
        return {tag: tag for tag in unique_tags}


if __name__ == "__main__":
    process_posts()
