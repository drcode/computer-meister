# Overview
A ".plan" file sets up a plan for acquiring information from a website. The goal is to see if we are able to retrieve desired information from a website, modifying the plan as needed until we can do it successfully.

# Commands

## target site "url"
This is the site we'll visit to answer the question

## login_required
This website requires a login to answer the query. The app should first check if the user is already logged in with the latest session state, otherwise display a browser page for the user to log in.

## enable_text_entry
Explicit command to say that text entry by the computer use agent is allowed

## explore_website_openai "exploration command" max_command_steps
Explore the website with openai computer use with the given computer use api instructions, populating the artifacts folder with information.

## click 920 33
simulates mouse click at the given location, always uses the left button.

## type "text to type into website"
enters text into the website (only allowed if `enable_text_entry` is already activated)

## wait 500
waits the given number of miliseconds

## vscroll -200
vertically scrolls on the page: A negative number means to scroll up and reveal more info at the bottom of the page.

## answer_query_images "query instructions"
answer the query using the 4 most recent screenshots from the website

## answer_query_text "query instructions"
answer the query use the 4 most recent page html artifacts for the site&query in the artifacts folder for this session.

# example 1
```plan
target_site "https://finance.yahoo.com"
explore_website_openai "bring up information on the stock AMZN"
answer_query "get the most recent price of AMZN"
```
