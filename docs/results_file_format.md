# Overview
For every attempt at a web retrieval, a `{sectionid}_results.md` file is generated. That file will contain a header that lists the time the retrieval took (in miliseconds), whether it could be done fully headless, and whether the user had to help to log in to the site. Bellow that is a divider, and the actual pertinent information that was retrieved from the site, information that will hopefully directly answer the question. If the retrieved information begins with the word FAIL, it will instead contain error information.

# Example 1: Successful Query
```markdown
- time: 4320
- headless: True
- help: False
---
The price of AMZN is $340.20
```

# Example 2: Failed Query
```markdown
- time: 10439
- headless: False
- help: False
---
FAIL
Websocket Error $323223
```
