function cb = compactbit(b)
% 二进制（数组存储）转为十进制（数组）
% b = bits array
% cb = compacted string of bits (using words of 'word' bits)

[nSamples, nbits] = size(b);
nwords = ceil(nbits/8); %每八个字符划分为一个十进制
cb = zeros([nSamples nwords], 'uint8');

for j = 1:nbits
    w = ceil(j/8); %向上取整（正无穷方向）压缩bit->word.
    cb(:,w) = bitset(cb(:,w), mod(j-1,8)+1, b(:,j));
end
