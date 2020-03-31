{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import qualified Torch.NN as NN
import qualified Torch.Utils.Image as Im
import qualified Torch.Typed.Aux as Aux
import qualified Torch.DType as D
import qualified Torch.Tensor as D
import qualified Torch.Device as D
import qualified Torch.Functional as D

import Torch.Typed.NN (conv2d, Conv2dSpec(Conv2dSpec))
import Torch.Typed.Tensor (Tensor(UnsafeMkTensor))

import Codec.Picture (readImage)
import Data.Proxy (Proxy(Proxy))
import GHC.TypeLits


data SomeRGBShape where
  SomeRGBShape :: forall nbatch nwidth nheight.
                  (KnownNat nbatch, KnownNat nwidth, KnownNat nheight)
                  => Proxy '[nbatch, nwidth, nheight, 3] -> SomeRGBShape


someRGBShape :: [Int] -> SomeRGBShape
someRGBShape [nbatch, nwidth, nheight, nchan] =
    case someNatVal (fromIntegral nbatch) of
    Nothing -> error "nbatch has to be positive!"
    Just (SomeNat (Proxy :: Proxy nbatch)) ->
      case someNatVal (fromIntegral nwidth) of
        Nothing -> error "nwidth has to be positive!"
        Just (SomeNat (Proxy :: Proxy nwidth)) ->
          case someNatVal (fromIntegral nheight) of
            Nothing -> error "nheight has to be positive!"
            Just (SomeNat (Proxy :: Proxy nheight)) ->
              case nchan of
                3 -> SomeRGBShape $ Proxy @'[nbatch, nwidth, nheight, 3]
                _ -> error ("nchan has to be 3, not " <> show nchan)

withRGBTensor ::
  D.Tensor
  -> (  forall nbatch nwidth nheight.
      (KnownNat nbatch, KnownNat nwidth, KnownNat nheight)
      => Tensor '(D.CPU, 0) D.UInt8 '[nbatch, nwidth, nheight, 3]
      -> r )
  -> r
withRGBTensor untypedTensor f = case someRGBShape (D.shape untypedTensor) of
    (SomeRGBShape (Proxy :: Proxy '[nbatch, nwidth, nheight, 3])) -> f $ UnsafeMkTensor @'(D.CPU, 0) @D.UInt8 @'[nbatch, nwidth, nheight, 3] untypedTensor

run ::
  forall nbatch nwidth nheight.
  (KnownNat nbatch, KnownNat nwidth, KnownNat nheight,
   Aux.IsAtLeast (nwidth + 1) 3 (CmpNat (nwidth + 1) 3),
   Aux.IsAtLeast ((nwidth - 3) + 1) 1 (CmpNat ((nwidth - 3) + 1) 1),
   Aux.IsAtLeast (nheight + 1) 3 (CmpNat (nheight + 1) 3),
   Aux.IsAtLeast ((nheight - 3) + 1) 1 (CmpNat ((nheight - 3) + 1) 1))
  => Tensor '(D.CPU, 0) D.Float '[nbatch, 3, nwidth, nheight]
  -> IO (Tensor '(D.CPU, 0) D.Float '[nbatch, 1, (nwidth - 3) + 1, (nheight - 3) + 1])
run x = do
    let spec = Conv2dSpec @3 @1 @3 @3 @D.Float @'(D.CPU, 0)
    sam <- NN.sample spec
    return $ conv2d @'(1, 1) @'(0, 0) sam x

main :: IO ()
main = do
    im <- readImage "./examples/dynamic-unet-cnn/paris.jpg" >>= return . (either error id)
    imtensor <- Im.fromDynImage im >>= return . (either error id)
    print $ withRGBTensor imtensor run
