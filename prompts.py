POS_INFERENCE_PROMPT = """
(masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
{} 
emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
sharp focus, high budget, cinemascope, moody, epic, gorgeous
"""

NEG_INFERENCE_PROMPT = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""

TEST_GPT_PROMPT = """
Based on the video content description, you need to write five coherent scene descriptions to create a silent video. These five descriptions are independent, but there needs to be a connection between the five scenes. The five scene descriptions should include detailed scenario transitions (such as camera movement, background changes, and object movement). The camera movement should be smooth. Avoid drastic angle changes and transitions, such as shifting from a frontal view directly to a side view. You can add details and objects, but the five scenes must form a continuous story, which means repeated object descriptions and details may be omitted. Five scene descriptions should NOT differ too much. Ensure similarity to enable smooth transitions between scenes. If the description is brief, you can add details, but stay conservative, and only create simple, easily generated scenes. It's also acceptable for multiple scenes to share a higher degree of similarity. You need to accurately, objectively, and succinctly describe everything. The scene descriptions need to be concise. Do NOT add details unrelated to the video content description. Do NOT describe the atmosphere or speculate. Do NOT add scene titles, directly return five scene descriptions. 

The Video Content Description: 
_WHOLE_CAPTION_
"""