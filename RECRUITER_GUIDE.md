# Recruiter Guide

This guide explains how to test Auxilium as a recruiter without going through the full technical documentation.

## Demo data notice

The demo mailbox and indexed demo content contain fictitious technical data only.

## Suggested demo flow

Recruiters can test Auxilium quickly by using the demo mailbox and asking a few technical questions based on the indexed email threads.

The current demo mailbox includes six main topics:

1. Database performance issue
2. Redis upgrade
3. Kubernetes migration
4. Suspicious login / security investigation
5. API rate limits
6. PostgreSQL 16 testing

Suggested questions:

- What performance issue affected the customer dashboard?
- Which database table caused the slow dashboard queries?
- What index was recommended for the `customer_events` table?
- What improvement was observed after deploying the index to staging?
- What version of Redis was proposed for the upgrade?
- Was Redis 7.2 approved for production rollout?
- Why was the Kubernetes migration postponed?
- What was the new migration date for the reporting service?
- Was the suspicious login a real compromise?
- What caused the suspicious login alert?
- What are the current API rate limits by plan?
- What version of PostgreSQL is currently used in production?
- What benchmark improvements were observed with PostgreSQL 16?
- When is the PostgreSQL 16 production migration scheduled?

### More advanced questions

- Summarise the Redis upgrade discussion.
- Explain why the Kubernetes migration plan changed.
- List all approved production changes and their scheduled dates.
- Summarise all infrastructure-related upgrades mentioned in the demo mailbox.
- Which technical issues were investigated and what were their outcomes?

## Sign in

The application requires Microsoft sign-in before email ingestion can be used.

If you do not want to sign in with a personal Microsoft account, you can use the demo account:

- email: `demo.auxilium@outlook.com`
- password: `Auxilium#2026`

## Open Settings

After signing in, open the settings menu from the top-right corner by clicking the user account area.

Optional screenshot placeholder:

![Settings access placeholder](./docs/screenshots/recruiter-settings-entry.png)

## Configure email ingestion

Open `E-mails` in Settings.

Then:

- select the mailbox folders you want Auxilium to ingest
- typical choices are Inbox and Sent Items

This controls which Outlook folders are read and transformed into searchable content.

Optional screenshot placeholder:

![Email settings placeholder](./docs/screenshots/recruiter-email-settings.png)

## Configure local folders

Open `Folders` in Settings.

Then:

- add or enable the local directories you want Auxilium to index
- only authorised folders are used by the local retrieval pipeline

This is the right place to test the local-document side of the product.

Optional screenshot placeholder:

![Folder settings placeholder](./docs/screenshots/recruiter-folder-settings.png)

## Understand `Ingestion emails` and `Reindex`

The app exposes two different actions:

- `Ingestion emails`: fetches emails from the selected mailbox folders and adds them to the local index
- `Reindex`: rebuilds the full local index from both sources:
  - ingested emails
  - authorised local folders

Recommended testing flow:

1. choose your email folders
2. run `Ingestion emails`
3. if you also want local documents included, configure `Folders` and run `Reindex`

## Use the correct answer mode

For document-grounded answers, use the mode that relies on the local indexed knowledge base rather than a general-purpose answer alone.

This is the most relevant mode for evaluating whether Auxilium can:

- retrieve facts from emails
- retrieve facts from indexed local files
- answer using the project knowledge base instead of only the base model

## Example questions to ask

For emails:

- "Summarise the latest discussion about API throttling."
- "What was the recommendation regarding Redis version upgrades?"
- "Draft a reply to the PostgreSQL 16 testing thread."

For local documents:

- "What are the main points in this document?"
- "Find the section that mentions deployment risks."
- "Compare the recommendations in the local document with the recent email thread."

For mixed retrieval:

- "Do the emails and local documents agree on the migration plan?"
- "What information appears in the document but not in the email discussion?"
