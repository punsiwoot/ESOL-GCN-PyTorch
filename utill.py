import torch

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_validation_loss = float('inf')

    def __call__(self, train_loss, validation_loss):
        if train_loss > (self.best_validation_loss - self.min_delta):
            self.counter = 0
            self.best_validation_loss = validation_loss

        elif  self.best_validation_loss - self.min_delta > train_loss  :
            self.best_validation_loss = validation_loss
            self.counter += 1
            print(f'case valid better case {self.counter}')
            if self.counter >= self.tolerance:
                self.early_stop = True

        else : 
            print('something wrong')

def train(model, optimizer, loss_fn,validation_set ,training_set, device, epochs:int=1000, is_early_stop:bool = True, show_result_at:int = 0):
    earlystop = EarlyStopping(tolerance=7,min_delta=1)
    train_losses_epoch = []
    valid_losses_epoch = []
    stop_at_epoch = epochs
    for epoch in range(epochs):
        model = model.to(device)
        train_losses = []
        valid_losses = []
        
        #train step
        for batch in training_set:
            # Use GPU
            batch.to(device)
            # Reset gradients
            optimizer.zero_grad()
            # Passing the node features and the connection info
            pred  = model(batch.x.float(), batch.edge_index, batch.batch)
            # Calculating the loss and gradients
            train_loss = loss_fn(pred, batch.y)
            train_loss.backward()
            if torch.cuda.is_available() : float_train_loss = float(train_loss.cpu().detach().numpy().astype(float))
            else : float_train_loss = float(train_loss.detach().numpy().astype(float))
            train_losses.append(float_train_loss)
            # Update using the gradients
            optimizer.step()
        
        #valid step
        model.eval()
        for batch in validation_set:
            # Use GPU
            batch.to(device)
            # Passing the node features and the connection info
            pred = model(batch.x.float(), batch.edge_index, batch.batch)
            # Calculating the loss
            valid_loss = loss_fn(pred, batch.y)
            if torch.cuda.is_available(): float_valid_loss = float(valid_loss.cpu().detach().numpy().astype(float))
            else : float_valid_loss = float(valid_loss.detach().numpy().astype(float))
            valid_losses.append(float_valid_loss)
        
        #calculate average loss
        average_train_loss = sum(train_losses)/len(train_losses)
        average_valid_loss = sum(valid_losses)/len(valid_losses)
        train_losses_epoch.append(average_train_loss)
        valid_losses_epoch.append(average_valid_loss)
        if (show_result_at != 0) and ((epoch+1)%show_result_at == 0 ): print(f"at epoch : {epoch+1} train_loss = {average_train_loss}  valid_loss = {average_valid_loss}")
        if (is_early_stop):earlystop(train_loss=average_train_loss, validation_loss=average_valid_loss)
        if earlystop.early_stop and is_early_stop:
            stop_at_epoch = epoch+1
            print(f'result at {epoch+1} is {earlystop.early_stop}')
            break
    # test_result = test(model,loss_fn=loss_fn,device)
    return train_losses_epoch, valid_losses_epoch, stop_at_epoch

def test(model,loss_fn,test_set,device):
    test_loss_list = []
    for batch in test_set:
        batch.to(device)
        # Passing the node features and the connection info
        pred = model(batch.x.float(), batch.edge_index, batch.batch)
        # Calculating the loss
        test_loss = loss_fn(pred, batch.y)
        if torch.cuda.is_available(): float_test_loss = float(test_loss.cpu().detach().numpy().astype(float))
        else : float_test_loss = float(test_loss.detach().numpy().astype(float))
        test_loss_list.append(float_test_loss)
    average_train_loss = sum(test_loss_list)/len(test_loss_list)
    return average_train_loss


