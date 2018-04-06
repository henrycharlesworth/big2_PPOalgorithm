import numpy as np

def shuffle(array):
    i=0
    j=0
    temp = 0
    
    for i in range(array.size-1,0,-1):
        j = int(np.floor(np.random.random()*(i+1)))
        temp = array[i]
        array[i] = array[j]
        array[j] = temp
    return array

def isPair(hand):
    if hand.size != 2:
        return 0
    if np.ceil(hand[0]/4) == np.ceil(hand[1]/4):
        return 1
    else:
        return 0
        
def isThreeOfAKind(hand):
    if hand.size != 3:
        return 0
    if (np.ceil(hand[0]/4)==np.ceil(hand[1]/4)) and (np.ceil(hand[1]/4)==np.ceil(hand[2]/4)):
        return 1
    else:
        return 0
        
def isFourOfAKind(hand):
    if hand.size != 4:
        return 0
    if (np.ceil(hand[0]/4)==np.ceil(hand[1]/4)) and (np.ceil(hand[1]/4)==np.ceil(hand[2]/4)) and (np.ceil(hand[2]/4)==np.ceil(hand[3]/4)):
        return 1
    else:
        return 0
        
def isTwoPair(hand):
    if hand.size != 4:
        return 0
    if isFourOfAKind(hand):
        return 0
    hand.sort()
    if isPair(hand[0:2]) and isPair(hand[2:]):
        return 1
    else:
        return 0

def isStraightFlush(hand):
    if hand.size != 5:
        return 0
    hand.sort()
    if (hand[0] + 4 == hand[1]) and (hand[1]+4==hand[2]) and (hand[2]+4==hand[3]) and (hand[3]+4==hand[4]):
        return 1
    else:
        return 0
        
def isStraight(hand):
    if hand.size != 5:
        return 0
    hand.sort()
    if (np.ceil(hand[0]/4)+1==np.ceil(hand[1]/4)) and (np.ceil(hand[1]/4)+1==np.ceil(hand[2]/4)) and (np.ceil(hand[2]/4)+1==np.ceil(hand[3]/4)) and (np.ceil(hand[3]/4)+1==np.ceil(hand[4]/4)):
        return 1
    else:
        return 0
        
def isFlush(hand):
    if hand.size != 5:
        return 0
    if (hand[0] % 4 == hand[1] % 4) and (hand[1] % 4 == hand[2] % 4) and (hand[2] % 4 == hand[3] % 4) and (hand[3] % 4 == hand[4] % 4):
        return 1
    else:
        return 0

#returns the value of the 3 card part        
def isFullHouse(hand):
    if hand.size != 5:
        return (False,)
    hand.sort()
    if isPair(hand[0:2]) and isThreeOfAKind(hand[2:]):
        return (True, np.ceil(hand[3]/4))
    elif isThreeOfAKind(hand[0:3]) and isPair(hand[3:]):
        return (True, np.ceil(hand[0]/4))
    else:
        return (False,)
        
def isRealHand(hand):
    if (hand.size > 5) or (hand.size < 1):
        return 0
    if hand.size==1:
        return 1
    if hand.size==2:
        if isPair(hand):
            return 1
        else:
            return 0
    if hand.size==3:
        if isThreeOfAKind(hand):
            return 1
        else:
            return 0
    if hand.size==4:
        if isTwoPair(hand):
            return 1
        elif isFourOfAKind(hand):
            return 1
        else:
            return 0
    if hand.size==5:
        if isStraight(hand):
            return 1
        elif isFlush(hand):
            return 1
        elif isFullHouse(hand):
            return 1
        else:
            return 0
        
        
def validatePlayedHand(hand, prevHand, control):
    if not isRealHand(hand):
        return 0
    if control==1:
        return 1 #can play any real hand with control
    if hand.size != prevHand.size:
        return 0 #must be same size if not in control
    
    hand.sort()
    prevHand.sort()
    
    if hand.size == 1:
        if hand[0] > prevHand[0]:
            return 1
        else:
            return 0
    elif hand.size == 2:
        if not isPair(hand):
            return 0
        else:
            if hand[1] > prevHand[1]:
                return 1
            else:
                return 0
    elif hand.size == 3:
        if not isThreeOfAKind(hand):
            return 0
        else:
            if hand[2] > prevHand[2]:
                return 1
            else:
                return 0
    elif hand.size == 4:
        if isFourOfAKind(hand):
            if not isFourOfAKind(prevHand):
                return 1
            else:
                if hand[3] > prevHand[3]:
                    return 1
                else:
                    return 0
        if isTwoPair(hand):
            if isFourOfAKind(prevHand):
                return 0
            else:
                if hand[3] > prevHand[3]:
                    return 1
                else:
                    return 0
        return 0
    elif hand.size == 5:
        
        if isStraightFlush(hand):
            if not isStraightFlush(prevHand):
                return 1
            else:
                if hand[4] > prevHand[4]:
                    return 1
                else:
                    return 0
        
        fh = isFullHouse(hand)
        if fh[0] == True:
            if isStraightFlush(prevHand):
                return 0
            fhph = isFullHouse(prevHand)
            if fhph[0] == False:
                return 1
            else:
                if fh[1] > fhph[1]:
                    return 1
                else:
                    return 0
                
        if isFlush(hand):
            if isFullHouse(prevHand)[0]:
                return 0
            elif isStraightFlush(prevHand):
                return 0
            if not isFlush(prevHand):
                return 1
            else:
                if hand[4] > prevHand[4]:
                    return 1
                else:
                    return 0
                
        if isStraight(hand):
            if isFullHouse(prevHand)[0]:
                return 0
            elif isFlush(prevHand):
                return 0
            elif isStraightFlush(prevHand):
                return 0
            
            if hand[4] > prevHand[4]:
                return 1
            else:
                return 0
            
    

#function to convert hand in text form into number form.
def convertHand(hand):
    #takes a list in the form ["3H","KD",...] etc and converts it into numbers
    output = np.zeros(len(hand))
    counter = 0
    for card in hand:
        if card[0] == "2":
            base = 13
        elif card[0] == "A":
            base = 12
        elif card[0] == "K":
            base = 11
        elif card[0] == "Q":
            base = 10
        elif card[0] == "J":
            base = 9
        elif card[0] == "1":
            base = 8
            card = card.replace("0","")
        else:
            base = int(card[0])-2
        
        if card[1] == "D":
            suit = 1
        elif card[1] == "C":
            suit = 2
        elif card[1] == "H":
            suit = 3
        elif card[1] == "S":
            suit = 4
            
        output[counter] = int((base-1)*4 + suit)
        counter += 1
    return output





#we need a function that evaluates an initial hand and evaluates all of the hands which are available.
#so have a vector of 2 card hands, 3 card hands, etc. We should then have a function which updates all of the available hands
#when a particular hand is played.

def cardValue(num):
    return np.ceil(num/4)

class card:
    def __init__(self, number, i):
        self.suit = number % 4 #1 - Diamond, 2 - Club, 3- Heart, 0 - Spade
        self.value = np.ceil(number/4) #from 1 to 13.
        self.indexInHand = i #index within current hand (from 0 to 12)
        self.inPair = 0
        self.inThreeOfAKind = 0
        self.inFourOfAKind = 0
        self.inFlush = 0
        self.inStraight = 0
        self.straightIndex = -1 #index of which straight this card is in.
        self.flushIndex = -1
        
    def __repr__(self):
        if self.value < 8:
            string1 = str(self.value+2)
            string1 = string1[0]
        elif self.value == 8:
            string1 = "10"
        elif self.value == 9:
            string1 = "J"
        elif self.value == 10:
            string1 = "Q"
        elif self.value == 11:
            string1 = "K"
        elif self.value == 12:
            string1 = "A"
        elif self.value == 13:
            string1 = "2"
        if self.suit == 1:
            string2 = "D"
        elif self.suit == 2:
            string2 = "C"
        elif self.suit == 3:
            string2 = "H"
        else:
            string2 = "S"
        cardString = string1 + string2
        return "<card. %s inPair: %d, inThree: %d, inFlush: %d, inStraight: %d>" % (cardString, self.inPair, self.inThreeOfAKind, self.inFlush, self.inStraight)
        

class handsAvailable:
    def __init__(self, currentHand, nC=0):
        self.cHand =  np.sort(currentHand).astype(int)
        self.handLength = currentHand.size
        self.cards = {}
        for i in range(self.cHand.size):
            self.cards[self.cHand[i]] = card(self.cHand[i],i)
        self.flushes = []
        self.pairs = []
        self.threeOfAKinds = []
        self.fourOfAKinds = []
        self.straights = []
        self.nPairs = 0
        self.nThreeOfAKinds = 0
        self.nDistinctPairs = 0
        if nC == 2:
            self.fillPairs()
        elif nC == 3:
            self.fillThreeOfAKinds()
        elif nC == 4:
            self.fillFourOfAKinds()
            self.fillPairs()
        else:
            self.fillPairs()
            self.fillSuits()
            self.fillStraights()
            self.fillThreeOfAKinds()
            self.fillFourOfAKinds()
    def fillSuits(self):
        self.diamonds = np.zeros((self.handLength,))
        self.clubs = np.zeros((self.handLength,))
        self.hearts = np.zeros((self.handLength,))
        self.spades = np.zeros((self.handLength,))
        dc = 0
        cc = 0
        hc = 0
        sc = 0
        for i in range(self.handLength):
            val = self.cHand[i] % 4
            if val == 1:
                self.diamonds[dc] = self.cHand[i]
                dc += 1
            elif val == 2:
                self.clubs[cc] = self.cHand[i]
                cc += 1
            elif val == 3:
                self.hearts[hc] = self.cHand[i]
                hc += 1
            else:
                self.spades[sc] = self.cHand[i]
                sc += 1
        self.diamonds = self.diamonds[0:dc]
        self.clubs = self.clubs[0:cc]
        self.hearts = self.hearts[0:hc]
        self.spades = self.spades[0:sc]
        if self.diamonds.size >= 5:
            self.flushes.append(self.diamonds)
        if self.clubs.size >= 5:
            self.flushes.append(self.clubs)
        if self.hearts.size >= 5:
            self.flushes.append(self.hearts)
        if self.spades.size >= 5:
            self.flushes.append(self.spades)
        for i in range(len(self.flushes)):
            flushes = self.flushes[i]
            for j in range(flushes.size):
                self.cards[flushes[j]].inFlush = 1
                self.cards[flushes[j]].flushIndex = i
    
    def fillStraights(self):
        streak = 0
        cInd = 0
        sInd = 0
        while cInd < self.cHand.size - 1:
            cVal = self.cards[self.cHand[cInd]].value
            nVal = self.cards[self.cHand[cInd+1]].value
            if nVal == cVal + 1:
                streak += 1
                cInd += 1
            elif nVal == cVal:
                cInd += 1
            else:
                if streak >= 4:
                    self.straights.append(self.cHand[sInd:cInd+1])
                streak = 0
                cInd = cInd + 1
                sInd = cInd
        if streak >= 4:
            self.straights.append(self.cHand[sInd:])
        for i in range(len(self.straights)):
            straight = self.straights[i]
            for j in range(straight.size):
                self.cards[straight[j]].inStraight = 1
                self.cards[straight[j]].straightIndex = i
                
    def fillPairs(self):
        cVal = -1
        nDistinct = 0
        for i in range(self.handLength-1):
            for j in range(i+1,i+4):
                if j>=self.handLength:
                    continue
                if isPair(np.array([self.cHand[i], self.cHand[j]])):
                    nVal = cardValue(self.cHand[i])
                    if nVal != cVal:
                        nDistinct += 1
                        cVal = nVal
                    self.pairs.append([self.cHand[i], self.cHand[j]])
                    self.nPairs += 1
                    self.nDistinctPairs = nDistinct
                    self.cards[self.cHand[i]].inPair = 1
                    self.cards[self.cHand[j]].inPair = 1
                    
    def fillThreeOfAKinds(self):
        for i in range(self.handLength-2):
            for j in range(i+1,i+3):
                if (j+1)>=self.handLength:
                    continue
                if isThreeOfAKind(np.array([self.cHand[i], self.cHand[j], self.cHand[j+1]])):
                    self.threeOfAKinds.append([self.cHand[i], self.cHand[j], self.cHand[j+1]])
                    self.nThreeOfAKinds += 1
                    self.cards[self.cHand[i]].inThreeOfAKind = 1
                    self.cards[self.cHand[j]].inThreeOfAKind = 1
                    self.cards[self.cHand[j+1]].inThreeOfAKind = 1
                    
    def fillFourOfAKinds(self):
        for i in range(self.handLength-3):
            if self.cards[self.cHand[i]].suit==1:
                if np.ceil(self.cHand[i]/4) == np.ceil(self.cHand[i+1]/4):
                    if np.ceil(self.cHand[i]/4) == np.ceil(self.cHand[i+2]/4):
                        if np.ceil(self.cHand[i]/4) == np.ceil(self.cHand[i+3]/4):
                            self.fourOfAKinds.append([self.cHand[i], self.cHand[i+1], self.cHand[i+2], self.cHand[i+3]])
                            self.cards[self.cHand[i]].inFourOfAKind = 1
                            self.cards[self.cHand[i+1]].inFourOfAKind = 1
                            self.cards[self.cHand[i+2]].inFourOfAKind = 1
                            self.cards[self.cHand[i+3]].inFourOfAKind = 1            
        
        
    