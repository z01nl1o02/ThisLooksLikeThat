# ThisLooksLikeThat

# paper
[This looks like that: deep learning for interpretable image recognition](https://arxiv.org/abs/1806.10574)

by Chaofan Chen, Oscar Li, Alina Barnett, Jonathan Su, Cynthia Rudin


# usage

## demo_cifar2.py
   input train.lst/test.lst, output trained net and proto.pkl
## show.prototype.py
   parse proto.pkl and save prototype as image

## CMakeLists.txt
   some codes implemented by cpu codes to save 


# test
  * results/cifar-29400.params   
    batch size = 20 train for 29400 interation, test accuracy 69%
  * results prototxt     
  ![prototypes](https://github.com/z01nl1o02/ThisLooksLikeThat/blob/master/results/canvas.png)
