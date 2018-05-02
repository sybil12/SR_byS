function cb = compactbit(b)
% �����ƣ�����洢��תΪʮ���ƣ����飩
% b = bits array
% cb = compacted string of bits (using words of 'word' bits)

[nSamples, nbits] = size(b);
nwords = ceil(nbits/8); %ÿ�˸��ַ�����Ϊһ��ʮ����
cb = zeros([nSamples nwords], 'uint8');

for j = 1:nbits
    w = ceil(j/8); %����ȡ�����������ѹ��bit->word.
    cb(:,w) = bitset(cb(:,w), mod(j-1,8)+1, b(:,j));
end
