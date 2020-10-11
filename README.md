

# Purpose
KeyExtract allows users ranging from students to researchers to extract useful information from big texts or essays. By a simple click of a button, the webapp returns a list of words that met your % requirement and also correlated to your word/ sentence of choice.

# Inspiration
Due to being in COVID-19, online school has really hurt my vision by forcing me to read long articles and spending hours finding that one simple detail. That's why me and Bryan came up with the idea to use the tensorflow model we came across to build the webapp.

# How we built it
Utilizing Flask framework, tensorflow models and some html, css frontend, we pieced together the webapp. The model encodes the sentences, allowing us to find the cosine similarity from the sentences which equates to a similarity %. The flask app then checks the set % and sees if the similarity is higher and then outputting it through a form request and then ends up on the frontend.

# Challenges we ran into
The challenges we ran into was the tensorflow session, we realized that the tensorflow model could only be utilized in the session because of some code mistakes. However after consulting a mentor, we quickly identified the problem and solved it. Another challenge we ran into was getting the html to work, Bryan's first time using html was impressive but did require some skill honing, so we both worked on it together.

# What we enjoyed
We enjoyed the great games in Sunhacks and the process of developing our very own webapp. Bryan especially enjoyed the air force challenge as he loves cryptography. My first hackathon was a blast, hopefully many more to come!

# What's next for KeyExtract
We're planning on switching to a chrome extension to allow ease of access between your work and KeyExtract. We also are looking to make it paid so users can donate to charity while accessing the benefit of precision extracting.
