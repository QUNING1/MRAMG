proposals = [propose(llm, query) for _ in range(3)]

votes = []
for agent in [text_agent, visual_agent, judge_agent]:
    votes.append(agent.vote(proposals))

selected_proposal = aggregate_votes(votes)

text_answer = text_agent.answer(query, selected_proposal)
visual_answer = visual_agent.answer(query, selected_proposal)