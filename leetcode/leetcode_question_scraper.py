import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm  # Progress bar
import re

# URL of the webpage
URL = "https://bishalsarang.github.io/Leetcode-Questions/out.html"

# List of LeetCode Top 150 question titles
leetcode_top_150 = [
    "Merge Sorted Array", 
    "Remove Element", 
    "Remove Duplicates from Sorted Array", 
    "Remove Duplicates from Sorted Array II", 
    "Majority Element",
    "Rotate Array",
    "Best Time to Buy and Sell Stock",
    "Best Time to Buy and Sell Stock II",
    "Jump Game",
    "Jump Game II",
    "H-Index",
    "Insert Delete GetRandom O(1)",
    "Product of Array Except Self",
    "Gas Station",
    "Candy",
    "Trapping Rain Water",
    "Roman to Integer",
    "Integer to Roman",
    "Length of Last Word",
    "Longest Common Prefix",
    "Reverse Words in a String",
    "Zigzag Conversion",
    "Find the Index of the First Occurrence in a String",
    "Text Justification",
    "Valid Palindrome",
    "Is Subsequence",
    "Two Sum II - Input Array Is Sorted",
    "Container With Most Water",
    "3Sum",
    "Minimum Size Subarray Sum",
    "Longest Substring Without Repeating Characters",
    "Substring with Concatenation of All Words",
    "Minimum Window Substring",
    "Valid Sudoku",
    "Spiral Matrix",
    "Rotate Image",
    "Set Matrix Zeroes",
    "Game of Life",
    "Ransom Note",
    "Isomorphic Strings",
    "Word Pattern",
    "Valid Anagram",
    "Group Anagrams",
    "Two Sum",
    "Happy Number",
    "Contains Duplicate II",
    "Longest Consecutive Sequence",
    "Summary Ranges",
    "Merge Intervals",
    "Insert Interval",
    "Minimum Number of Arrows to Burst Balloons",
    "Valid Parentheses",
    "Simplify Path",
    "Min Stack",
    "Evaluate Reverse Polish Notation",
    "Basic Calculator",
    "Linked List Cycle",
    "Add Two Numbers",
    "Merge Two Sorted Lists",
    "Copy List with Random Pointer",
    "Reverse Linked List II",
    "Reverse Nodes in k-Group",
    "Remove Nth Node From End of List",
    "Remove Duplicates from Sorted List II",
    "Rotate List",
    "Partition List",
    "LRU Cache",
    "Maximum Depth of Binary Tree",
    "Same Tree",
    "Invert Binary Tree",
    "Symmetric Tree",
    "Construct Binary Tree from Preorder and Inorder Traversal",
    "Construct Binary Tree from Inorder and Postorder Traversal",
    "Populating Next Right Pointers in Each Node II",
    "Flatten Binary Tree to Linked List",
    "Path Sum",
    "Sum Root to Leaf Numbers",
    "Binary Tree Maximum Path Sum",
    "Binary Search Tree Iterator",
    "Count Complete Tree Nodes",
    "Lowest Common Ancestor of a Binary Tree",
    "Binary Tree Right Side View",
    "Average of Levels in Binary Tree",
    "Binary Tree Level Order Traversal",
    "Binary Tree Zigzag Level Order Traversal",
    "Minimum Absolute Difference in BST",
    "Kth Smallest Element in a BST",
    "Validate Binary Search Tree",
    "Number of Islands",
    "Surrounded Regions",
    "Clone Graph",
    "Evaluate Division",
    "Course Schedule",
    "Course Schedule II",
    "Snakes and Ladders",
    "Minimum Genetic Mutation",
    "Word Ladder",
    "Implement Trie (Prefix Tree)",
    "Design Add and Search Words Data Structure",
    "Word Search II",
    "Letter Combinations of a Phone Number",
    "Combinations",
    "Permutations",
    "Combination Sum",
    "N-Queens II",
    "Generate Parentheses",
    "Word Search",
    "Convert Sorted Array to Binary Search Tree",
    "Sort List",
    "Construct Quad Tree",
    "Merge k Sorted Lists",
    "Maximum Subarray",
    "Maximum Sum Circular Subarray",
    "Search Insert Position",
    "Search a 2D Matrix",
    "Find Peak Element",
    "Search in Rotated Sorted Array",
    "Find First and Last Position of Element in Sorted Array",
    "Find Minimum in Rotated Sorted Array",
    "Median of Two Sorted Arrays",
    "Kth Largest Element in an Array",
    "IPO",
    "Find K Pairs with Smallest Sums",
    "Find Median from Data Stream",
    "Add Binary",
    "Reverse Bits",
    "Number of 1 Bits",
    "Single Number",
    "Single Number II",
    "Bitwise AND of Numbers Range",
    "Palindrome Number",
    "Plus One",
    "Factorial Trailing Zeroes",
    "Sqrt(x)",
    "Pow(x, n)",
    "Max Points on a Line",
    "Climbing Stairs",
    "House Robber",
    "Word Break",
    "Coin Change",
    "Longest Increasing Subsequence",
    "Triangle",
    "Minimum Path Sum",
    "Unique Paths II",
    "Longest Palindromic Substring",
    "Interleaving String",
    "Edit Distance",
    "Best Time to Buy and Sell Stock III",
    "Best Time to Buy and Sell Stock IV",
    "Maximal Square"
]

# Output directory
OUTPUT_FOLDER = "LeetCode_Top_150_Questions"

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_title(raw_title):
    """
    Removes leading numbers and periods from the title.
    Example: "7. Reverse Integer" -> "Reverse Integer"
    """
    return re.sub(r"^\d+\.\s*", "", raw_title).strip()

def scrape_questions(url):
    """
    Scrapes the webpage and extracts only the questions that match `leetcode_top_150`.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise error if request fails

    # Parse HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Dictionary to store extracted questions
    questions_dict = {}

    # Find all title divs
    title_divs = soup.find_all("div", id="title")

    # Progress bar for processing titles
    for title_div in tqdm(title_divs, desc="Checking Titles", unit="title"):
        raw_title_text = title_div.get_text(strip=True)
        cleaned_title = clean_title(raw_title_text)  # Remove numbers from title

        # Check if the cleaned title matches `leetcode_top_150`
        if cleaned_title in leetcode_top_150:
            # Find the next div containing the question content
            question_div = title_div.find_next("div", class_="content__u3I1 question-content__JfgR")

            if question_div:
                question_text = question_div.get_text("\n", strip=True)
                questions_dict[cleaned_title] = question_text

    return questions_dict

def save_questions(questions):
    """
    Saves extracted questions as individual text files.
    """
    for title in tqdm(questions, desc="Saving Questions", unit="file"):
        file_name = f"{title.replace(' ', '_')}.txt"
        file_path = os.path.join(OUTPUT_FOLDER, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(questions[title])

    print(f"\n✅ Successfully saved {len(questions)} questions in '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    extracted_questions = scrape_questions(URL)  # Scrape matching questions
    save_questions(extracted_questions)  # Save as text files