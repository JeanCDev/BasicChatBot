/// DOC: https://withcatai.github.io/node-llama-cpp/guide/CUDA
/// Cuda compilation error fix: https://github.com/NVlabs/tiny-cuda-nn/issues/164#issuecomment-1280749170
/// Compile chatbot using CUDA:  npx --no node-llama-cpp download --cuda


import {LlamaCpp} from "@langchain/community/llms/llama_cpp";
import { ChatLlamaCpp } from "@langchain/community/chat_models/llama_cpp";
//import { HumanMessage } from "@langchain/core/messages";

const llamaPath = "./codeninja-1.0-openchat-7b.Q8_0.gguf";



// Max achieved prompt - default config
// const prompt = `Create a resume of this text: Most models can be cut into pieces and split across different hardware that combined still work as the original. Its a bit like a group assignment. You might be able to give every person in the group a different part of it so that when you are all done and combine your work you have the end result you were assigned to create. If you choose minus one you choose to give the GPU (the fastest person of the group) all the work and let the others do nothing. And if that fastest person can handle the amount of work that is the best option for Kobold because of how fast those GPU's are. But the moment you give it so much that it has no space to fit that work in the whole thing crashes. That does mean there is no solid answer to how many layers you need to put on what since that depends on your hardware. The first step is figuring out how much VRAM your GPU actually has. Once you know that you can make a reasonable guess how many layers you can put on your GPU. We list the required size on the menu. So for example if you see a model that mentions 8GB of VRAM you can only put -1 if your GPU also has 8GB of VRAM (in some cases windows and other software also swallows VRAM up so if its a tight fit you may still need la few ayers on the CPU). If it has less you can typically figure out the layers by how much less it has. For example if you have 8GB of VRAM and you want to load a model that requires 16GB you would need at least half the layers on the CPU. The final thing you need to play around with is the maxtokens that you wish to allow versus the speed. The more of the model you put on the GPU the faster things will be. But if you don't have much space left of the story you won't be able to give it as much information at the time`;

// const prompt = `Most models can be cut into pieces and split across different hardware that combined still work as the original. Its a bit like a group assignment. You might be able to give every person in the group a different part of it so that when you are all done and combine your work you have the end result you were assigned to create. If you choose minus one you choose to give the GPU (the fastest person of the group) all the work and let the others do nothing. And if that fastest person can handle the amount of work that is the best option for Kobold because of how fast those GPU's are. But the moment you give it so much that it has no space to fit that work in the whole thing crashes. That does mean there is no solid answer to how many layers you need to put on what since that depends on your hardware. The first step is figuring out how much VRAM your GPU actually has. Once you know that you can make a reasonable guess how many layers you can put on your GPU. We list the required size on the menu. So for example if you see a model that mentions 8GB of VRAM you can only put -1 if your GPU also has 8GB of VRAM (in some cases windows and other software also swallows VRAM up so if its a tight fit you may still need la few ayers on the CPU). If it has less you can typically figure out the layers by how much less it has. For example if you have 8GB of VRAM and you want to load a model that requires 16GB you would need at least half the layers on the CPU. The final thing you need to play around with is the maxtokens that you wish to allow versus the speed. The more of the model you put on the GPU the faster things will be. But if you don't have much space left of the story you won't be able to give it as much information at the time. Making it forget sooner about things that happened earlier in the story. This is most noticable if you are playing a story and suddenly it stops generating because it is out of memory. In that case you can choose to either lower the max tokens slider until it works. Or you can choose less layers on the GPU to free up that extra space for the story. You will have to toy around with it to find what you like. If you try to put the model entirely on the CPU keep in mind that in that case the ram counts double since the techniques we use to half the ram only work on the GPU. But if you have a lot of RAM with very little VRAM this can be worth it.`;
const prompt = `What was the last thing i asked you?`; // Model can't 'remember' last message

const testeLlamaCppInvoke = async(prompt) => {
  const model = new LlamaCpp({
    modelPath: llamaPath,
    maxTokens: 2048, // response tokens
    temperature: 0.7,
    gpuLayers: 512,
    cache: true,
    batchSize: 2048, // input tokens
  });

  console.time();
  console.log(`You: ${prompt}`);
  const response = await model.invoke(prompt);
  console.log(`AI : ${response}`);
  console.timeEnd();
}

const testeChatLlamaCpp = async(prompt) => {
  const model = new ChatLlamaCpp({
    modelPath: llamaPath,
    axTokens: 2048, // response tokens
    temperature: 0.7,
    gpuLayers: 512,
    cache: true,
    batchSize: 2048, // input tokens
  });

  console.time();
  console.log(`You: ${prompt}`);
  const response = await model.invoke(['imageurl', ...prompt])
/* }); */
  console.log(response.content);
  console.timeEnd();
}

//testeLlamaCppInvoke(prompt)

testeChatLlamaCpp([
  'tell me a joke',
  'and give an example of basic python code'
]); // Multiple inputs