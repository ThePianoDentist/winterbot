create table picks (
id SERIAL PRIMARY KEY,
match bigint references matches(matchID),
isRadiant boolean NOT NULL,
isPick boolean NOT NULL,
pickBanNum integer NOT NULL,
heroID integer NOT NULL
);

create table matches (
matchID bigint PRIMARY KEY,
radiantTeamID integer NOT NULL,
direTeamID integer NOT NULL
);
