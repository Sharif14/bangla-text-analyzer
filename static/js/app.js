// উদাহরণ (React/Next.js থেকে):
fetch("https://bangla-text-analyzer-yjkp.onrender.com/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: "...." }),
  })
    .then(res => res.json())
    .then(data => console.log(data));