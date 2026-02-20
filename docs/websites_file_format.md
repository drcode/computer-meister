# Overview
A `websites.md` file defines website-query targets in a markdown-friendly, line-parsable format.

Each section represents exactly one query + one site reference.

# Section Format
Each section header must use:

`# <site> <unique_number>`

Rules:
- `<site>` is a single hostname/domain string (no protocol, such as `twitter.com`).
- `<unique_number>` is an integer.
- If the same site appears multiple times, each occurrence must use a different number.

# Section Body
Each section must contain:
- Exactly one query line describing the desired information to retrieve.

Optional lines:
- `- disabled` to mark the section as disabled.
- Comments using HTML comment syntax: `<!-- comment text -->`.

# Parsing Rules
- One section per query target.
- One site per section.
- One query line per section.
- URLs must be host/domain only (no `http://` or `https://`).
- Blank lines may appear between sections.
- Comment lines (`<!-- ... -->`) should be ignored by parsers.

# Example (`websites.md`)
```markdown
# twitter.com 0
- disabled
get the ten most recent notifications
# weather.com 0
get the temperature in Mill Valley, CA
# finance.yahoo.com 0
- disabled
find the most recent price of AMZN
# etrade.com 0
- disabled
find the amount of stock holdings in the "Conrad" portfolio, also the cash balance for the account.
# twitter.com 1
- disabled
get the ten most recent tweets in my feed
```
